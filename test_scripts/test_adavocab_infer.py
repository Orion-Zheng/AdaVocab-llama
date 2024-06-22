import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from adavocab_llama.ada_vocab_factory import AdaVocabLlamaForCausalLM, AdaVocabGemmaforCausalLM, AdaVocabQwen2ForCausalLM
import warnings
warnings.filterwarnings("ignore")

def infer_test(model, device):
    chat = [
        { "role": "user", "content": "Who are you?" },
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, 
                       add_special_tokens=False, 
                       return_tensors='pt')
    inputs = inputs.to(device)
    pred = model.generate(**inputs,
                          max_new_tokens=64,
                          do_sample=False,
                        #   top_k=50,
                          num_return_sequences=1)
    print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=False))

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # model_path = "google/gemma-2b-it"
    # adavocab_model_path = "experiment_ckpts/Ada_Gemma-2b-FT-2024-06-20-215925/checkpoint-1449"  # 1024 intermediate dim
    # adavocab_model = AdaVocabGemmaforCausalLM.from_pretrained(adavocab_model_path, device_map=device)

    model_path = "Qwen/Qwen2-1.5B-Instruct"
    adavocab_model_path = "experiment_ckpts/Ada_qwen2-1.5b-FT-2024-06-19-154935/checkpoint-1526"  # 768 intermediate dim
    adavocab_model = AdaVocabQwen2ForCausalLM.from_pretrained(adavocab_model_path, torch_dtype=torch.float16, device_map=device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    adavocab_model.eval()
    # adavocab_model.offload_lm_head()
    torch.cuda.reset_peak_memory_stats()
    print('===== Output from AdaVocab Model =====')
    infer_test(adavocab_model, device)
    print("======================================")
    print(f"AdaVocab Model Maximum GPU memory usage: {torch.cuda.max_memory_allocated() / 1024**2} MB")
    del adavocab_model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.float16)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    print('===== Output from Original Model =====')
    infer_test(model, device)
    print("======================================")
    print(f"Original Model Maximum GPU memory usage: {torch.cuda.max_memory_allocated() / 1024**2} MB")