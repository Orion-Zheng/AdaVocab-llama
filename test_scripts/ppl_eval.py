import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from adavocab_llama.ada_vocab_factory import AdaVocabLlamaForCausalLM, AdaVocabGemmaforCausalLM, AdaVocabQwen2ForCausalLM

device = "cuda"
# model_path = "google/gemma-7b"
model_path = "Qwen/Qwen2-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)

adavocab_model_path = "experiment_ckpts/Ada_qwen2-1.5b-FT-2024-06-19-154935/checkpoint-1526"  # 768 intermediate dim
model = AdaVocabQwen2ForCausalLM.from_pretrained(adavocab_model_path, torch_dtype=torch.float16, device_map=device)

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer(test["text"], add_special_tokens=False) # use tokenizer parallelism
encodings.input_ids = torch.tensor([sum(encodings.input_ids, [])])  # flatten the list of lists

max_length = 4096 
stride = 2048

seq_len = encodings.input_ids.size(1)
model.eval()
nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    # input_ids[:, 0] = 2 # bos token for gemma
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)
    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print("Perplexity:", ppl)

# Gemma-2b: 7.7730
# Gemma-7b: 6.1210

# 