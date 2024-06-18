import os
import sys
import math
import torch
import numpy as np
import random
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (Trainer, AutoModelForCausalLM,
                          PreTrainedTokenizer,
                          DataCollatorForLanguageModeling,
                          AutoConfig, TrainerCallback,
                          LlamaForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, load_from_disk, load_metric
from accelerate import Accelerator, init_empty_weights
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union, Sequence
from transformers.integrations.integration_utils import WandbCallback
from datetime import datetime, timedelta

from codebase.monkey_patch import new_wandb_on_train_end, SafeSavingCallback
from codebase.utils import print_trainable_parameters, load_tokenizer, prepare_for_train, enable_flash_attn, set_model_config
from codebase.args_parser import parse_args
from codebase.dist_logging import get_dist_logger

from adavocab_llama.ada_vocab_factory import AdaVocabLlamaForCausalLM, AdaVocabGemmaforCausalLM, AdaVocabQwen2ForCausalLM
from adavocab_llama.ada_trainer import AdaTrainer
# can skip if you have already logged in at console by 'wandb login'
# import wandb
# wandb.login(key="")
# wandb.init(project='', name='')

logger = get_dist_logger()
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['WANDB_LOG_MODEL'] = 'true'
IGNORE_INDEX = -100
SAFE_MINUTES = 5

SafeSavingCallback.safe_minutes = SAFE_MINUTES

@dataclass
class AdaVocabArgs():
    ADA_DIM: int
    ADA_TOPK: int
    ADA_LOSS_WEIGHT: float
    ADA_MASK_WEIGHT: float
    ADA_TOPK_WEIGHT: float
    ADA_ACT: bool = False
    ADA_DORA: bool = False
    ADA_SVD: bool = False

@dataclass
class PaddToMaxLenCollator(object):
    # Adapt from: https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/6e8c6c23e51ec8f0cf8a2b1f1633e52edb768e9c/scripts/training/build_dataset.py
    tokenizer: PreTrainedTokenizer
    max_len: int
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Truncate each sequence to `max_len`. If dataset is already packed to max_len, no truncation will be done.
        input_ids, labels = tuple([torch.tensor(instance[key])[:self.max_len] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def load_model(model_args, quant_config=None, peft_config=None, model_config=None, model_class=AutoModelForCausalLM):
    kwargs = {
        "torch_dtype": eval(f"torch.{model_args.load_dtype}"),
        "config": model_config  # e.g. you want to reuse the ckpt but change the config
    }
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    
    if model_args.use_flash:
        enable_flash_attn(model_args.model_dir)

    model = model_class.from_pretrained(model_args.model_dir, **kwargs)
    if model_args.do_train:
        model = prepare_for_train(model, model_args)  # e.g. disable kv cache, freeze some modules(specified in the model_args)
    if quant_config:
        # 1- Cast the layernorm in fp32; 2- making output embedding layer require grads; 3- Add the upcasting of the lm_head to fp32
        model = prepare_model_for_kbit_training(model)
    if peft_config:
        model = get_peft_model(model, peft_config)

    return model

def enable_monkey_patch():
    logger.info('New `on_train_end` is applied to `WandbCallback`')
    WandbCallback.on_train_end = new_wandb_on_train_end


def main():
    enable_monkey_patch() 
    custom_args = [AdaVocabArgs]
    model_args, data_args, trainer_config, peft_config, quant_config, log_args, other_args = parse_args(*custom_args)
    # TODO: add option to not using eval data(considering training arguments)
    logger.info("Load Training Dataset ...")
    train_data = load_from_disk(data_args.train_data_dir)
    logger.info("Load Evaluation Dataset ...")
    eval_data = load_from_disk(data_args.eval_data_dir) if data_args.eval_data_dir else None
    if data_args.input_column:
        train_data = train_data.rename_column(data_args.input_column, "input_ids")
        eval_data = eval_data.rename_column(data_args.input_column, "input_ids") if data_args else None
        
    logger.info(f"Training Data:\n{train_data}")
    logger.info(f"Evaluation Data:\n{eval_data}")
    
    tokenizer = load_tokenizer(model_args.tokenizer_dir, train_mode=model_args.do_train)
    
    # == Add AdaVocab HyperParam to model config before loading model ==
    model_config = AutoConfig.from_pretrained(model_args.model_dir)
    ada_config_dict = asdict(other_args[0])
    model_config = set_model_config(model_config, ada_config_dict)
    ADA_TOPK = model_config.ADA_TOPK
    logger.info(f"Final Model Config:\n{model_config}")
    # ===================================================================
    
    model = load_model(model_args, quant_config, peft_config, model_config, AdaVocabGemmaforCausalLM)
    
    logger.info(f"Model Architecture:\n{model}")
    print_trainable_parameters(model)
    
    def compute_metrics(eval_preds):
        """
        Sum the metrics of all batches and return the average.
        eval_preds.shape: (num_batch, ) * num of metrics
        """
        (token_accuracy, 
        mask_hit_rate, 
        top_k_diff, 
        mask_top_1_hit_rate, 
        mask_top_5_hit_rate,
        mask_top_10_hit_rate,
        mask_top_20_hit_rate,
        lm_loss, 
        mask_loss, 
        topk_loss) = eval_preds.predictions
        return {'token_accuracy': token_accuracy.mean().item(), 
                'mask_hit_rate': mask_hit_rate.mean().item(), 
                'top_k_diff': top_k_diff.mean().item(),
                'mask_top_1_hit_rate': mask_top_1_hit_rate.mean().item(),
                'mask_top_5_hit_rate': mask_top_5_hit_rate.mean().item(),
                'mask_top_10_hit_rate': mask_top_10_hit_rate.mean().item(),
                'mask_top_20_hit_rate': mask_top_20_hit_rate.mean().item(), 
                'lm_loss': lm_loss.mean().item(),
                'mask_loss': mask_loss.mean().item(),
                'topk_loss': topk_loss.mean().item(),
                }
    
    def get_topk_logits(logits, topk):
        # logits: (bs * seq_len, vocab_size)
        # get the topk largest values and their indices, top_k_indices: (bs * seq_len, topk)
        _, topk_indices = logits.topk(topk, dim=1)

        topk_logits = torch.zeros_like(logits)
        ones = torch.ones_like(topk_indices, dtype=torch.float)

        # set the topk_indices to 1, others to 0, in `vocab_size` dimension
        topk_logits.scatter_(1, topk_indices, ones)
        return topk_logits
    
    def get_token_level_mask_hit_rate(ada_logits_topk, lm_head_logits_topk, top_k):
        product = lm_head_logits_topk * ada_logits_topk

        # count the number of 1s at each position
        # sum along the vocab_size dimension and keep the dimension unchanged
        hit_count_tensor = product.sum(dim=-1, keepdim=True)  
        hit_rate_tensor = hit_count_tensor / top_k
        
        return hit_rate_tensor.mean()
    
    def count_token_level_positive(ada_logits_viewed):
        # greater_than_zero: (bs * seq_len, vocab_size)
        # greater_than_zero --> positive result after applying the sigmoid
        greater_than_zero = ada_logits_viewed > 0

        # count the number of positive values at each position along the vocab_size dimension
        count_greater_than_zero = greater_than_zero.int().sum(dim=1, keepdim=True)
        return count_greater_than_zero.float().mean()
    
    def calculate_token_accuracy(shift_labels, shift_logits_argmax):
        assert shift_labels.shape == shift_logits_argmax.shape, 'shift_labels and shift_logits_argmax should have the same torch size'

        # compare the elements of two tensors and calculate the number of equal elements
        equal_elements = shift_labels.eq(shift_logits_argmax)
        num_equal_elements = equal_elements.sum()
        # get the number of total elements
        total_elements = shift_labels.numel()
        # calculate the ratio of equal elements
        equal_ratio = num_equal_elements / total_elements
        return equal_ratio

    def preprocess_logits_for_metrics(logits, labels):
        """
        Add preprocess_logits_for_metrics
        
        logits.keys(): ['logits', 'lm_head_logits', 'lm_loss', 'mask_loss', 'topk_loss']
        
        """
        ada_logits = logits[0]       # torch.Size([2, 2048, 32000]) (bs, seq_len, vocab_size)
        lm_head_logits = logits[1]   # torch.Size([2, 2048, 32000])
        lm_loss = logits[2] 
        mask_loss = logits[3]
        topk_loss = logits[4]
        assert ada_logits.shape == lm_head_logits.shape, "ada_logits and lm_head_logits should have the same shape."
        
        bs = ada_logits.shape[0]
        seq_len = ada_logits.shape[1]
        vocab_size = ada_logits.shape[2]
        
        token_accuracy = None # SHIFT token_level
        mask_hit_rate = None   # token_level
        top_k_diff = None   #  token_level
        mask_top_1_hit_rate = None # token_level    
        
        # mask_hit_rate
        ada_logits_topk = get_topk_logits(ada_logits.view(bs * seq_len, vocab_size), ADA_TOPK) # (bs * seq_len, vocab_size)
        lm_head_logits_topk = get_topk_logits(lm_head_logits.view(bs * seq_len, vocab_size), ADA_TOPK) # (bs * seq_len, vocab_size) ,  torch.unique(lm_head_logits_topk): tensor([0., 1.], device='cuda:0')
        mask_hit_rate = get_token_level_mask_hit_rate(ada_logits_topk, lm_head_logits_topk, ADA_TOPK) # token_level

        # top_1_hit_rate
        lm_head_logits_top1 = get_topk_logits(lm_head_logits.view(bs * seq_len, vocab_size), 1)
        mask_top_1_hit_rate = get_token_level_mask_hit_rate(ada_logits_topk, lm_head_logits_top1, top_k=1)
        
        # top_5_hit_rate
        lm_head_logits_top5 = get_topk_logits(lm_head_logits.view(bs * seq_len, vocab_size), 5)
        mask_top_5_hit_rate = get_token_level_mask_hit_rate(ada_logits_topk, lm_head_logits_top5, top_k=5)
        
        # top_10_hit_rate
        lm_head_logits_top10 = get_topk_logits(lm_head_logits.view(bs * seq_len, vocab_size), 10)
        mask_top_10_hit_rate = get_token_level_mask_hit_rate(ada_logits_topk, lm_head_logits_top10, top_k=10)
        
        # top_20_hit_rate
        lm_head_logits_top20 = get_topk_logits(lm_head_logits.view(bs * seq_len, vocab_size), 20)
        mask_top_20_hit_rate = get_token_level_mask_hit_rate(ada_logits_topk, lm_head_logits_top20, top_k=20)
        
        # top_k_diff
        top_k_diff = count_token_level_positive(ada_logits.view(bs * seq_len, vocab_size)) - ADA_TOPK # token_level
        
        # token_accuracy
        shift_logits = ada_logits[..., :-1, :].contiguous().view(-1, vocab_size)  # (batch_size * (seq_len - 1), vocab_size)
        shift_logits_argmax = torch.argmax(shift_logits, dim=1).view(bs * (seq_len - 1))
        shift_labels = labels[..., 1:].contiguous().view(-1)
        token_accuracy = calculate_token_accuracy(shift_labels, shift_logits_argmax)
        
        return (token_accuracy, 
                mask_hit_rate, 
                top_k_diff, 
                mask_top_1_hit_rate, 
                mask_top_5_hit_rate,
                mask_top_10_hit_rate,
                mask_top_20_hit_rate,
                lm_loss, 
                mask_loss, 
                topk_loss)
    
    trainer = AdaTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data, 
        args=trainer_config,
        data_collator=PaddToMaxLenCollator(tokenizer, model_args.max_length), 
        compute_metrics=compute_metrics, 
        preprocess_logits_for_metrics=preprocess_logits_for_metrics, 
        callbacks=[SafeSavingCallback]  # Add SafeSavingCallback to save model when time is running out
    )
    
    # Training
    if model_args.do_train:
        train_result, model_at_end = trainer.train(resume_from_checkpoint=model_args.resume_from_checkpoint)
        trainer.save_final_checkpoint(model_at_end)
        trainer.log_metrics("train", train_result.metrics)  # e.g {'train_runtime': 112.9526, 'train_samples_per_second': 0.142, 'train_steps_per_second': 0.035, 'train_loss': 9.430782318115234, 'epoch': 0.0, 'num_input_tokens_seen': 14166}
        trainer.save_metrics("train", train_result.metrics)  # all_results.json

    # Evaluation
    if model_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")  # e.g {'eval_loss': 8.76622486114502, 'eval_runtime': 9.493, 'eval_samples_per_second': 10.534, 'eval_steps_per_second': 5.267, 'epoch': 0.0, 'num_input_tokens_seen': 14166}
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")

        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)  # eval_results.json

if __name__ == "__main__":
    main()