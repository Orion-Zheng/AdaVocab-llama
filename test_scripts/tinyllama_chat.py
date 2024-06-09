from transformers import AutoModelForCausalLM, AutoTokenizer
from adavocab_llama.ada_vocab_factory import AdaVocabLlamaForCausalLM
import warnings
warnings.filterwarnings("ignore")

import torch

def gpu_mem_test(model):
    input = "Who are "
    inputs = tokenizer(input, return_tensors='pt')
    inputs = inputs.to('cuda:0')
    pred = model.generate(**inputs,
                      max_new_tokens=256,
                      do_sample=False,
                      top_k=50,
                      top_p=0.95,
                      num_return_sequences=1)
    # print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[len(input):])

if __name__ == "__main__":
    # model_path = "/mnt/vepfs/lczza/failed-llama2-model"
    model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    model = AdaVocabLlamaForCausalLM.from_pretrained(pretrained_model_name_or_path = "/net/papilio/storage7/tingyuan/llama/ckpt/final_ckpt_backup-1901", device_map = "auto")
    if model.training == False:
        model.offload_lm_head()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    torch.cuda.reset_peak_memory_stats()
    gpu_mem_test(model)
    print(f"Adatinyllama Maximum GPU memory usage: {torch.cuda.max_memory_allocated() / 1024**2} MB")
    model = model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    torch.cuda.reset_peak_memory_stats()
    gpu_mem_test(model)
    print(f"tinyllama Maximum GPU memory usage: {torch.cuda.max_memory_allocated() / 1024**2} MB")