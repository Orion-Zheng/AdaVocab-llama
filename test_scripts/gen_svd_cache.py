from transformers import AutoModelForCausalLM
from adavocab_llama.ada_vocab_factory import create_factorized_compression_for_linear

model_paths = ["experiment_ckpts/gemma-2b_SFT-2024-06-10-123619/checkpoint-11592",
               "experiment_ckpts/qwen2-1.5b_SFT-2024-06-10-125011/checkpoint-12212"]
for model_path in model_paths:
    model = AutoModelForCausalLM.from_pretrained(model_path)
    _ = create_factorized_compression_for_linear(model.lm_head, rank=128)