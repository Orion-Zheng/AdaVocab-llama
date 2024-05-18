import math
import warnings
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.llama.modeling_llama import LlamaModel, LlamaPreTrainedModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache


ADA_RATIO = 4
ADA_TOPK = 20
ADA_LOSS_WEIGHT = 0.1 # lm_loss: 10.375   mask_loss: 0.6931
ADA_TOPK_WEIGHT = 0.00000005 # topk_loss: 32727040
# ADA_LOSS_WEIGHT * lm_loss + mask_loss + ADA_TOPK_WEIGHT * topk_loss

@dataclass
class AdaCausalLMOutputWithPast(CausalLMOutputWithPast):
    # keep original `loss` for `training_step` and `predictions_step`, 
    # Add 3 sub losses: `lm_loss`, `mask_loss`, `topk_loss`
    # add `lm_head_logits` for original lm_head logits, which is optional (required for train and eval, not required for generation)
    lm_head_logits: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    mask_loss: Optional[torch.FloatTensor] = None
    topk_loss: Optional[torch.FloatTensor] = None


class AdaVocabHead(nn.Module):  # The same as LoRALayer
    def __init__(self, hidden_size, vocab_size, sub_vocab_dim):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(sub_vocab_dim).float())
        # TODO: Consider adding non-linear activation function
        # TODO: Investigate the rationale of parameter initialization  Tingyuan: Maybe consider using random initialization with low variance
        self.A = nn.Parameter(torch.randn(hidden_size, sub_vocab_dim) * std_dev)
        self.B = nn.Parameter(torch.zeros(sub_vocab_dim, vocab_size))

    def forward(self, x):
        # x.shape: (..., hidden_size), A.shape: (hidden_size, sub_vocab_dim), B.shape: (sub_vocab_dim, vocab_size)
        ada_vocab_logits = x @ self.A @ self.B  # ada_vocab_logits.shape: (..., vocab_size)
        return ada_vocab_logits

    
class AdaVocabLlamaForCausalLM(LlamaForCausalLM):  # For Training(train with LM Head)
    # TODO: Check the function of this variable and if it affects the AdaVocab Head model
    _tied_weights_keys = ["lm_head.weight"]  

    def __init__(self, config):
        super().__init__(config)
        self.sub_vocab_dim = config.vocab_size // ADA_RATIO  
        self.topK = ADA_TOPK
        # AdaVocabHead is already initialized with random weights, 
        # so no need to use `self.post_init` method after this
        self.adavocab_head = AdaVocabHead(config.hidden_size, 
                                          config.vocab_size, 
                                          self.sub_vocab_dim)
        self.freeze_original_model()
    
    def freeze_original_model(self):
        # freeze orginal llama except AdaVocabHead
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False
        for param in self.adavocab_head.parameters():
            param.requires_grad = True

    def topk_mask(self, logits):
        # logits.shape: (batch_size, seq_len, vocab_size)
        topk_values, topk_indices = torch.topk(logits, self.topK, dim=-1)
        # topk_values.shape, topk_indices.shape: (batch_size, seq_len, topK)
        mask = torch.zeros_like(logits)  # (batch_size, seq_len, vocab_size)
        # Only in top-k positions, put 1 to the corresponding position
        mask.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(mask))

        return mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # TODO: How does forward know whether is training or inference?
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # hidden_states.shape: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, _ = hidden_states.size()
       
        # This activation could be very large during training if vocab_size is large,
        # but in inference, storing activation is not needed
        ada_logits = self.adavocab_head(hidden_states)  # (batch_size, seq_len, vocab_size)  
        ada_logits = ada_logits.float()
        
        lm_head_logits = None
        lm_loss, mask_loss, topk_loss = None, None, None

        loss = None
        if labels is not None:  # prediction_step, training_step. Not for generation
            # ------ Only for Training ------
            # During Inference, we don't need self.lm_head in GPU memory
            lm_head_logits = self.lm_head(hidden_states)   # (batch_size, seq_len, vocab_size)  
            lm_head_logits = lm_head_logits.float()
            # -------------------------------
            # Supervised Signal of `self.adavocab_head` from two sources: 
            # 1. (Primary) BCEWithLogitsLoss between ada_logits and topk_gt_mask (distillation signal)
            # 2. CrossEntropyLoss between ada_logits and labels with constraint (from ground truth labels)
            
            # Loss from the first source
            # Shift so that tokens < n predict n
            shift_logits = ada_logits[..., :-1, :].contiguous()  # (batch_size, seq_len - 1, vocab_size)
            shift_labels = labels[..., 1:].contiguous()  # (batch_size, seq_len - 1)

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()  # CE loss includes the softmax function
            shift_logits = shift_logits.view(-1, self.config.vocab_size)  # (batch_size * (seq_len - 1), vocab_size)

            shift_labels = shift_labels.view(-1)  # (batch_size * seq_len)
            shift_labels = shift_labels.to(shift_logits.device)
            
            lm_loss = loss_fct(shift_logits, shift_labels)
            
            # Loss from the second source
            ada_logits_flat = ada_logits.view(-1, self.config.vocab_size)  # (batch_size * seq_len, vocab_size)
            ada_probs = torch.sigmoid(ada_logits_flat)  # (batch_size * seq_len, vocab_size)
            
            topk_gt_mask = self.topk_mask(lm_head_logits)  # (batch_size, seq_len, vocab_size)
            topk_gt_mask = topk_gt_mask.view(-1, self.config.vocab_size)  # (batch_size * seq_len, vocab_size)
            
            mask_loss_fct = BCEWithLogitsLoss()  # BCE Loss including the sigmoid function
            mask_loss = mask_loss_fct(ada_logits_flat, topk_gt_mask)

            ada_ones = ada_probs.sum()  # scalar
            # TODO: Pad Token Handle
            target_ones = batch_size * seq_len * self.topK  # scalar  
            target_ones = torch.tensor(target_ones, dtype=torch.float32).to(ada_ones.device)
            topk_loss = F.l1_loss(ada_ones, target_ones)

            loss = ADA_LOSS_WEIGHT * lm_loss + mask_loss + ADA_TOPK_WEIGHT * topk_loss

        if not return_dict:
            output = (ada_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return AdaCausalLMOutputWithPast(
            loss=loss,
            logits=ada_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            # New added
            lm_head_logits=lm_head_logits,
            lm_loss=lm_loss,
            mask_loss=mask_loss,
            topk_loss=topk_loss,
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    # TODO: Add `get` and `set` methods for `adavocab_head`