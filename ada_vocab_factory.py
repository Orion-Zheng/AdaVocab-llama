import math
import warnings
import hashlib
import os
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.llama.modeling_llama import LlamaModel, LlamaPreTrainedModel, LlamaForCausalLM
from transformers.models.gemma.modeling_gemma import GemmaModel, GemmaPreTrainedModel, GemmaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2PreTrainedModel, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache

from codebase.dist_logging import get_dist_logger
logger = get_dist_logger()

def svd_with_cache(matrix, cache_dir, max_rank=1024):
    """
    SVD with cache mechanism to avoid repeated SVD computation.
    SVD can be very slow for large matrices, so we cache the results.
    """
    in_dim, out_dim = matrix.shape
    # slice_weight = matrix[::1000, :]  # too sensitive to precision
    # weight_hash = hashlib.md5(slice_weight.detach().cpu().numpy().tobytes()).hexdigest()
    weight_hash = in_dim * out_dim  
    cache_file = os.path.join(cache_dir, f'{weight_hash}.pt')

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if os.path.exists(cache_file):
        # Load cached SVD results
        logger.info(f"Loading cached SVD results from {cache_file}")
        U, S, Vh = torch.load(cache_file)
    else:
        # Perform SVD and cache the results
        logger.info(f'Performing SVD and save to cache {cache_file}')
        U, S, Vh = torch.linalg.svd(matrix.float())
        U = U[:, :max_rank].clone()  # Shape: [out_features, rank]
        S = S[:max_rank].clone()     # Shape: [rank]
        Vh = Vh[:max_rank, :].clone()  # Shape: [rank, in_features]
        # Save the SVD results to cache
        torch.save((U, S, Vh), cache_file)
    return U, S, Vh

def create_factorized_compression_for_linear(source_linear, rank, svd_cache_dir='experiment_cache/'):
    """
    Adapt from: https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/cli_svd.py
    Create a factorized compression for a given linear layer using SVD.
    Args:
        source_linear (nn.Linear): The original linear layer to be compressed.
        rank (int, optional): The rank for the factorization. If None, it will be calculated based on rank_factor.
        rank_factor (float, optional): The factor to determine the rank if rank is not provided. Default is 0.3.
    Returns:
        nn.Sequential: A sequential container of the compressed linear layers.
    """
    logger.info(f"Creating SVD AdaHead with rank {rank}")
    with torch.no_grad():
        dtype = source_linear.weight.dtype
        # Check if the source linear layer has a bias term
        if hasattr(source_linear, 'bias'):
            bias = source_linear.bias
        else:
            bias = None
        # Calculate the total number of parameters in the source linear layer
        source_num_params = sum(param.numel() for param in source_linear.parameters())
        # Get the weight matrix of the source linear layer
        source_linear_weight = source_linear.weight.data
        # Ensure rank is less than the minimum dimension of the weight matrix
        assert rank < min(source_linear_weight.shape)
        # Perform SVD on the weight matrix
        # U, S, Vh = torch.linalg.svd(source_linear_weight.float())
        U, S, Vh = svd_with_cache(source_linear_weight, svd_cache_dir)
        # Truncate U, S, Vh to the specified rank
        U = U[:, :rank].contiguous()  # Shape: [out_features, rank]
        S = S[:rank].contiguous()     # Shape: [rank]
        Vh = Vh[:rank, :].contiguous()  # Shape: [rank, in_features]
        # Incorporate singular values into U
        U = U @ torch.diag(S)  # Shape: [out_features, rank]
        # Flatten U and Vh for quantile computation
        U_flatten = U.flatten()
        Vh_flatten = Vh.flatten()
        # Define the maximum quantization size
        max_quant_size = 2**23
        # Compute high and low quantile values for clamping
        if len(U_flatten) + len(Vh_flatten) >= max_quant_size:
            dist2 = U_flatten[:min(len(U_flatten), max_quant_size)]
            dist3 = Vh_flatten[:min(len(Vh_flatten), max_quant_size)]
            hi_val = max(torch.quantile(dist3, 1), torch.quantile(dist2, 1))
        else:
            dist = torch.cat([U_flatten, Vh_flatten])
            hi_val = torch.quantile(dist, 1)
        low_val = -hi_val
        # Clamp U and Vh to the quantile values
        U = U.clamp(low_val, hi_val)
        Vh = Vh.clamp(low_val, hi_val)
        # Create the down projection linear layer (Vh)
        lora_down = nn.Linear(Vh.shape[1], Vh.shape[0], dtype=dtype, bias=False, device=source_linear_weight.device)
        lora_down.weight.data = Vh.to(device=source_linear_weight.device, dtype=dtype)
        # Create the up projection linear layer (U)
        lora_up = nn.Linear(U.shape[1], U.shape[0], dtype=dtype, bias=bias is not None, device=source_linear_weight.device)
        lora_up.weight.data = U.to(device=source_linear_weight.device, dtype=dtype)
        # If the original linear layer had a bias, copy it to the up projection layer
        if bias is not None:
            lora_up.bias = nn.Parameter(bias.clone())
        # Print compression ratio (for debugging purposes)
        #print('compression', sum(param.numel() for param in ret.parameters()) / source_num_params)
        return lora_down, lora_up
    

@dataclass
class AdaCausalLMOutputWithPast(CausalLMOutputWithPast):
    # keep original `loss` for `training_step` and `predictions_step`, 
    # Add 3 sub losses: `lm_loss`, `mask_loss`, `topk_loss`
    # add `lm_head_logits` for original lm_head logits, which is optional (required for train and eval, not required for generation)
    lm_head_logits: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    mask_loss: Optional[torch.FloatTensor] = None
    topk_loss: Optional[torch.FloatTensor] = None

class AdaVocabHead_MLP(nn.Module):
  # No improvement compare to LoRA solution
  def __init__(self, lm_head, sub_vocab_dim, activation_func=torch.nn.GELU()):
    hidden_size, vocab_size = lm_head.in_features, lm_head.out_features
    super().__init__()

    self.A = nn.Linear(hidden_size, sub_vocab_dim, bias=False)
    self.B = nn.Linear(sub_vocab_dim, sub_vocab_dim, bias=True)
    self.C = nn.Linear(sub_vocab_dim, vocab_size, bias=False)
    std_dev = 1 / math.sqrt(sub_vocab_dim)
    nn.init.normal_(self.A.weight, 0, std_dev)
    nn.init.normal_(self.B.weight, 0, std_dev)
    nn.init.zeros_(self.C.weight)
    self.activation_func = activation_func
    
  def forward(self, x):
    # x.shape: (..., hidden_size), 
    # A.shape: (hidden_size, sub_vocab_dim)
    # B.shape: (sub_vocab_dim, sub_vocab_dim)
    # C.shape: (sub_vocab_dim, vocab_size)
    logits = self.A(x)  # logits.shape: (..., sub_vocab_dim)
    logits = self.activation_func(logits)  
    logits = self.B(logits)  # logits.shape: (..., sub_vocab_dim)
    # logits = self.activation_func(logits) 
    ada_vocab_logits = self.C(logits)  # ada_vocab_logits.shape: (..., vocab_size)  

    return ada_vocab_logits

class AdaVocabHead_LORA(nn.Module):
  def __init__(self, lm_head, sub_vocab_dim, svd=False):
    hidden_size, vocab_size = lm_head.in_features, lm_head.out_features
    super().__init__()
    if svd: # SVD initialization
      self.A, self.B = create_factorized_compression_for_linear(lm_head, sub_vocab_dim)
    else:  # Random initialization
      self.A = nn.Linear(hidden_size, sub_vocab_dim, bias=False)
      self.B = nn.Linear(sub_vocab_dim, vocab_size, bias=False)
      std_dev = 1 / math.sqrt(sub_vocab_dim)
      nn.init.normal_(self.A.weight, 0, std_dev)
      nn.init.zeros_(self.B.weight)
    
  def forward(self, x):
    # x.shape: (..., hidden_size), A.shape: (hidden_size, sub_vocab_dim), B.shape: (sub_vocab_dim, vocab_size)
    logits = self.A(x)
    ada_vocab_logits = self.B(logits)  # ada_vocab_logits.shape: (..., vocab_size)  
    return ada_vocab_logits

def create_AdaVocabCausalLM(base_class):  # Support LLama, Qwen2, Gemma 
    class AdaVocabCausalLM(base_class):  
        # TODO: Check the function of this variable and if it affects the AdaVocab Head model
        _tied_weights_keys = ["lm_head.weight"]  

        def __init__(self, config):
            super().__init__(config)
            self.sub_vocab_dim = config.ADA_DIM
            # AdaVocabHead is already initialized with random weights/ SVD weights
            # so no need to use `self.post_init` method after this
            if config.ADA_ACT:
                self.adavocab_head = AdaVocabHead_MLP(self.lm_head, self.sub_vocab_dim, activation_func=nn.GELU())
            else:
                self.adavocab_head = AdaVocabHead_LORA(self.lm_head, self.sub_vocab_dim, svd=config.ADA_SVD)

            self.freeze_original_model()
        
        def freeze_original_model(self):
            # freeze orginal llama except AdaVocabHead
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.lm_head.parameters():
                param.requires_grad = False
            for param in self.adavocab_head.parameters():
                param.requires_grad = True
        
        def offload_lm_head(self):
            self.lm_head = self.lm_head.to(torch.device('cpu'))
            
        def topk_mask(self, logits):
            # logits.shape: (batch_size, seq_len, vocab_size)
            topk_values, topk_indices = torch.topk(logits, self.config.ADA_TOPK, dim=-1)
            # topk_values.shape, topk_indices.shape: (batch_size, seq_len, topK)
            mask = torch.zeros_like(logits)  # (batch_size, seq_len, vocab_size)
            # Only in top-k positions, put 1 to the corresponding position
            mask.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(mask))
            return mask
        
        def pred_with_sliced_lm_head(self, ada_logits, hidden_states, input_ids, labels=None, min_logit=-100):
            nll_loss = None
            # Limit activated tokens to ADA_TOPK during inference
            ada_logits_mask = self.topk_mask(ada_logits)  # (batch_size, seq_len, vocab_size)
            ada_logits = ada_logits * ada_logits_mask  # (batch_size, seq_len, vocab_size)
            
            batch_size, seq_len, vocab_size = ada_logits.size()
            ada_index_slice = torch.nonzero(ada_logits[:, -1, :] > 0, as_tuple=True)[-1]  # equivalent to `sigmoid(ada_logits) > 0.5`
            union_ada_index_slice = torch.unique(ada_index_slice).to(self.lm_head.weight.device)  # torch_size([union_size])
            sliced_lm_head_weight = self.lm_head.weight[union_ada_index_slice, :].contiguous()  # torch.Size([union_size, hidden_size])
            ada_logits_sliced = hidden_states @ sliced_lm_head_weight.T  # (batch_size, seq_len, union_size)
            
            # Create a tensor of all `-inf`s with shape (batch_size, seq_len, vocab_size)
            pred_lm_logits = torch.full((batch_size, seq_len, vocab_size), min_logit, 
                                        dtype=ada_logits_sliced.dtype, device=input_ids.device)
            # Use union_ada_index_slice for filling
            # Assign the value of ada_logits to the specified location of pred_lm_logits through broadcasting
            pred_lm_logits.scatter_(2, union_ada_index_slice.expand(batch_size, seq_len, -1).to(input_ids.device), 
                                    ada_logits_sliced)  # (batch_size, seq_len, vocab_size)
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = pred_lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits_flatten = shift_logits.view(-1, self.config.vocab_size)
                shift_labels_flatten = shift_labels.view(-1)

                shift_labels_flatten = shift_labels_flatten.to(shift_logits.device)
                nll_loss = loss_fct(shift_logits_flatten, shift_labels_flatten)
            return pred_lm_logits, nll_loss

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
            cache_position: Optional[torch.LongTensor] = None,  # TODO: check the effect of this new variable
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
            vocab_size = self.lm_head.weight.shape[0]
        
            # This activation could be very large during training if vocab_size is large,
            # but in inference, storing activation is not needed
            ada_logits = self.adavocab_head(hidden_states)  # (batch_size, seq_len, vocab_size)  
            ada_logits = ada_logits.float()
            
            lm_head_logits = None
            lm_loss, mask_loss, topk_loss = None, None, None
            loss = None

            if labels is not None:  # For prediction_step, training_step. Not for generation
                # ------ Only for Training and Eval Loop------
                # During Inference, we don't need self.lm_head in GPU memory
                lm_head_logits = self.lm_head(hidden_states)   # (batch_size, seq_len, vocab_size)  
                lm_head_logits = lm_head_logits.float()
                # -------------------------------
                # Supervised Signal of `self.adavocab_head` from two sources: 
                # 1. (Primary) BCEWithLogitsLoss between ada_logits and topk_gt_mask (distillation signal)
                # 2. CrossEntropyLoss between ada_logits and labels with constraint (from ground truth labels)
                
                if self.training:  # training_step
                    # Loss from the second source
                    # Shift so that tokens < n predict n
                    shift_logits = ada_logits[..., :-1, :].contiguous()  # (batch_size, seq_len - 1, vocab_size)
                    shift_labels = labels[..., 1:].contiguous()  # (batch_size, seq_len - 1)

                    # Flatten the tokens
                    loss_fct = CrossEntropyLoss()  # CE loss includes the softmax function
                    shift_logits = shift_logits.view(-1, self.config.vocab_size)  # (batch_size * (seq_len - 1), vocab_size)

                    shift_labels = shift_labels.view(-1)  # (batch_size * seq_len)
                    shift_labels = shift_labels.to(shift_logits.device)
                    
                    lm_loss = loss_fct(shift_logits, shift_labels)
                else:  # prediction_step
                    _, lm_loss = self.pred_with_sliced_lm_head(ada_logits, hidden_states, input_ids, labels, min_logit=-100)

                # Loss from the first source
                ada_logits_flat = ada_logits.view(-1, self.config.vocab_size)  # (batch_size * seq_len, vocab_size)
                ada_probs = torch.sigmoid(ada_logits_flat)  # (batch_size * seq_len, vocab_size)
                
                topk_gt_mask = self.topk_mask(lm_head_logits)  # (batch_size, seq_len, vocab_size)
                # TODO: Add weights from lm_head_logits
                topk_gt_mask = topk_gt_mask.view(-1, self.config.vocab_size)  # (batch_size * seq_len, vocab_size)
                
                mask_loss_fct = BCEWithLogitsLoss()  # BCE Loss including the sigmoid function
                mask_loss = mask_loss_fct(ada_logits_flat, topk_gt_mask)

                ada_ones = ada_probs.sum()  # scalar
                # TODO: Handle pad token in no-packing case in the future
                target_ones = batch_size * seq_len * self.config.ADA_TOPK  # scalar
                target_ones = torch.tensor(target_ones, dtype=torch.float32).to(ada_ones.device)
                # We need to normalize this loss, make it agnostic to batch size, seq_len, topK
                topk_loss = F.l1_loss(ada_ones, target_ones) / target_ones

                loss = self.config.ADA_LOSS_WEIGHT * lm_loss + self.config.ADA_MASK_WEIGHT * mask_loss + self.config.ADA_TOPK_WEIGHT * topk_loss
            else:  # For generation
                with torch.no_grad():
                    lm_head_logits = ada_logits
                    ada_logits, loss = self.pred_with_sliced_lm_head(ada_logits, hidden_states, input_ids, min_logit=-100)

            if not return_dict:
                output = (ada_logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return AdaCausalLMOutputWithPast(
                loss=loss,
                logits=ada_logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                # Added by AdaVocab
                lm_head_logits=lm_head_logits if lm_head_logits is not None else None,
                lm_loss=self.config.ADA_LOSS_WEIGHT * lm_loss if lm_loss is not None else None,
                mask_loss=self.config.ADA_MASK_WEIGHT * mask_loss if mask_loss is not None else None,
                topk_loss=self.config.ADA_TOPK_WEIGHT * topk_loss if topk_loss is not None else None,
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
    return AdaVocabCausalLM

AdaVocabLlamaForCausalLM = create_AdaVocabCausalLM(LlamaForCausalLM)
AdaVocabGemmaforCausalLM = create_AdaVocabCausalLM(GemmaForCausalLM)
AdaVocabQwen2ForCausalLM = create_AdaVocabCausalLM(Qwen2ForCausalLM)

