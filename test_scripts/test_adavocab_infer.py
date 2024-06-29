import time
import torch
import warnings

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.logging import disable_progress_bar

from adavocab_llama.ada_vocab_factory import AdaVocabLlamaForCausalLM, AdaVocabGemmaforCausalLM, AdaVocabQwen2ForCausalLM
warnings.filterwarnings("ignore")
disable_progress_bar()

def add_chat_template(test_input, tokenizer):
    chat = [
        { "role": "user", "content": test_input },
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return prompt

def infer_test(model, prompt, device):
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
    # print('INPUT: ', tokenizer.decode(inputs['input_ids'][0]))
    inputs = inputs.to(device)
    start_time = time.time()
    pred = model.generate(**inputs,
                          max_new_tokens=32,
                          do_sample=False,
                        #   top_k=50,
                          num_return_sequences=1)
    end_time = time.time()
    return tokenizer.decode(pred.cpu()[0], skip_special_tokens=False), end_time - start_time

if __name__ == "__main__":
    test_input = """Q: Who won Super Bowl XX?\n\nA:"""
    test_input = """What should I do to be a good researcher?"""
    
    device = torch.device('cuda')
    # device = torch.device('cpu')
    dtype = torch.float16
    print("======================================")
    print(f"Device: {device}; Data Type: {dtype}")
    
    # model_path = "experiment_ckpts/gemma-2b_SFT-2024-06-10-123619/checkpoint-11592"  # WildChat-SFT-ep2
    # adavocab_model_path = "experiment_ckpts/Ada_Gemma-2b-FT-2024-06-20-215925/checkpoint-1449"  # 1024 intermediate dim
    # adavocab_model_path = "experiment_ckpts/Ada_Gemma-2b-FT-2024-06-19-031619/checkpoint-1449"  # 512 intermediate dim
    # adavocab_model = AdaVocabGemmaforCausalLM.from_pretrained(adavocab_model_path, torch_dtype=dtype, device_map=device)

    model_path = "experiment_ckpts/qwen2-1.5b_SFT-2024-06-10-125011/checkpoint-12212"
    adavocab_model_path = "experiment_ckpts/Ada_qwen2-1.5b-FT-2024-06-19-154935/checkpoint-1526"  # 768 intermediate dim
    adavocab_model_path = "experiment_ckpts/Ada_qwen2-1.5b-FT-2024-06-19-060126/checkpoint-1526"  # 384 intermediate dim
    model_id = "experiment_ckpts/Ada_qwen2-1.5b-FT-2024-06-27-194032/checkpoint-2573"  # (pre-train) 384 intermediate dim
    adavocab_model = AdaVocabQwen2ForCausalLM.from_pretrained(adavocab_model_path, torch_dtype=dtype, device_map=device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # test_input = add_chat_template(test_input, tokenizer)
    adavocab_model.eval()
    # adavocab_model.offload_lm_head()
    torch.cuda.reset_peak_memory_stats()
    a_pred, a_elapsed_time = infer_test(adavocab_model, test_input, device)
    print("======================================")
    print(f"AdaVocab Model\n Elapsed time: {a_elapsed_time} seconds\n Maximum GPU memory usage: {torch.cuda.max_memory_allocated() / 1024**2} MB")
    del adavocab_model
    torch.cuda.empty_cache()
    
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=dtype)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    o_pred, o_elapsed_time = infer_test(model, test_input, device)
    print("======================================")
    print(f"Original Model\n Elapsed time: {o_elapsed_time} seconds\n Maximum GPU memory usage: {torch.cuda.max_memory_allocated() / 1024**2} MB")
    del model
    torch.cuda.empty_cache()
    print("======================================")
    print('===== Output from Original Model =====')
    print(f"{o_pred}")
    print('===== Output from AdaVocab Model =====')
    print(f"{a_pred}")