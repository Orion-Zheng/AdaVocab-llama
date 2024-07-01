import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from adavocab_llama.ada_vocab_factory import AdaVocabLlamaForCausalLM, AdaVocabGemmaforCausalLM, AdaVocabQwen2ForCausalLM
import matplotlib.pyplot as plt

def plot_label_ranking(y_1, y_2=None, mean=False): 
  # y_1.shape/y_2.shape: (num_seq, effective_seq_len)
  if mean:
    if len(y_1.shape) == 2:
      y_1 = y_1.mean(dim=0)
    if len(y_2.shape) == 2:
      y_2 = y_2.mean(dim=0)

  if len(y_1.shape) == 2:
    for i in range(y_1.size(0)):
        x = torch.arange(len(y_1[i]))
        plt.scatter(x.numpy(), y_1[i].numpy(), label=f'Row {i}', c='blue')
  else:
    x = torch.arange(len(y_1))
    plt.scatter(x.numpy(), y_1.numpy(), c='blue')

  if y_2 is not None: 
    if len(y_2.shape) == 2:
      for i in range(y_2.size(0)):
          x = torch.arange(len(y_2[i]))
          plt.scatter(x.numpy(), y_2[i].numpy(), label=f'Row {i}', c='red')
    else:
      x = torch.arange(len(y_2))
      plt.scatter(x.numpy(), y_2.numpy(), c='red')

  plt.xlabel('Position')
  plt.ylabel("Ranking")
  plt.title("Label's Ranking in Logits")
  plt.show()
  plt.savefig("label_ranking.png")

def get_label_rank_in_logits(lm_head_logits, labels, IGNORE_INDEX=-100):
    # shifted_lm_head_logits: (bs*seq_len - 1, vocab_size)
    # labels: (bs*seq_len - 1, )
    ids_not_ignore = ~(labels == IGNORE_INDEX)
    labels = labels[ids_not_ignore]
    lm_head_logits = lm_head_logits[ids_not_ignore]
    labels = labels.unsqueeze(1)
    ids_sort = torch.argsort(lm_head_logits, dim=-1, descending=True)
    label_rank = ids_sort.gather(-1, labels).flatten()
    return label_rank

def labels_not_in_top_k_logits(lm_head_logits, labels, k, IGNORE_INDEX=-100):
    # shifted_lm_head_logits: (bs*seq_len - 1, vocab_size)
    # labels: (seq_len - 1 , )
    top_k_indices = torch.topk(lm_head_logits, k, dim=-1).indices  # (bs*seq_len, k)
    label_ids_in_topk = (top_k_indices == labels.unsqueeze(1)).any(-1)  # (bs*seq_len, )
    label_ignore = (labels == IGNORE_INDEX)
    return labels[~label_ids_in_topk & ~label_ignore]

def labels_not_select_by_adavocab(ada_logits, labels, IGNORE_INDEX=-100):
    # shifted_ada_logits: (bs*seq_len - 1, vocab_size)
    # labels: (seq_len - 1, )
    ids_not_ignore = ~(labels == IGNORE_INDEX)
    labels = labels[ids_not_ignore]
    ada_logits = ada_logits[ids_not_ignore]
    adavocab_selected = (ada_logits > 0)
    label_selected_mask = adavocab_selected.gather(-1, labels.unsqueeze(1)).flatten()
    label_not_selected_mask = ~label_selected_mask
    return labels[label_not_selected_mask], label_not_selected_mask

device = "cuda"
model_path = "experiment_ckpts/gemma-2b_SFT-2024-06-10-123619/checkpoint-11592"
model_path = "experiment_ckpts/qwen2-1.5b_SFT-2024-06-10-125011/checkpoint-12212"
adavocab_model_path = "experiment_ckpts/Ada_qwen2-1.5b-FT-2024-06-19-154935/checkpoint-1526"  # 768 intermediate dim
adavocab_model_path = "experiment_ckpts/Ada_qwen2-1.5b-FT-2024-06-28-132733/checkpoint-2573"

tokenizer = AutoTokenizer.from_pretrained(model_path)
def detokenize(tokens):
    return tokenizer.convert_ids_to_tokens(list(tokens))
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
adavocab_model = AdaVocabQwen2ForCausalLM.from_pretrained(adavocab_model_path, torch_dtype=torch.float16, device_map=device)

test_set_text = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer(test_set_text["text"], add_special_tokens=False) # use tokenizer parallelism
total_input_ids = torch.tensor([sum(encodings['input_ids'], [])])

# test_set_tokenized = load_from_disk("tokenized_datasets/wildchat-1M_Qwen2_2048_sft_split/eval")
# total_input_ids = torch.tensor([sum(test_set_tokenized['input_ids'], [])])  # flatten the list of lists
IGNORE_INDEX = -100
max_length = 2048 
ignore_begin_tokens = 10  # ignore the loss on first 10 tokens
total_seq_len = total_input_ids.size(1)
adavocab_model.eval()
model.eval()
nlls = []

origin_fail_tokens = []
adavocab_fail_tokens = []

origin_label_ranks = []
adavocab_label_ranks = []

for begin_loc in tqdm(range(0, total_seq_len, max_length)):
    end_loc = begin_loc + max_length
    if end_loc >= total_seq_len:
        break
    input_ids = total_input_ids[:, begin_loc:end_loc].to(device)
    if tokenizer.bos_token_id is not None:
        input_ids[:, 0] = tokenizer.bos_token_id
    target_ids = input_ids.clone()
    # ignore loss on the first `ignore_begin_tokens` tokens to avoid extreme loss values
    target_ids[:, :ignore_begin_tokens] = -100 

    with torch.no_grad():
        origin_outputs = model(input_ids)
        origin_lm_head_logits = origin_outputs.logits
        bs, seq_len, vocab_size = origin_lm_head_logits.size()

        adavocab_outputs = adavocab_model(input_ids)
        adavocab_lm_head_logits = adavocab_outputs.logits
        ada_head_logits = adavocab_outputs.lm_head_logits
        ada_head_logits = adavocab_model.topk_mask(ada_head_logits) * ada_head_logits
        ada_head_logits_flatten = ada_head_logits[..., :-1, :].contiguous().view(-1, vocab_size)

        shift_labels_flatten = target_ids[..., 1:].contiguous().view(-1)
        adavocab_shift_lm_logits_flatten = adavocab_lm_head_logits[..., :-1, :].contiguous().view(-1, vocab_size)
        origin_shift_lm_logits_flatten = origin_lm_head_logits[..., :-1, :].contiguous().view(-1, vocab_size)
        
        ce_loss = nn.CrossEntropyLoss(reduction="none")
        origin_loss = ce_loss(origin_shift_lm_logits_flatten, shift_labels_flatten)  # position-wise loss
        adavocab_loss = ce_loss(adavocab_shift_lm_logits_flatten, shift_labels_flatten)  # position-wise loss
        
        fail_tokens_origin = labels_not_in_top_k_logits(origin_shift_lm_logits_flatten, shift_labels_flatten, adavocab_model.config.ADA_TOPK)
        fail_tokens_adavocab_gt, fail_tokens_pos = labels_not_select_by_adavocab(ada_head_logits_flatten, shift_labels_flatten)
        ids_not_ignore = ~(shift_labels_flatten == IGNORE_INDEX)
        ada_head_pred_tokens_greedy = adavocab_shift_lm_logits_flatten[ids_not_ignore].argmax(-1)
        fail_tokens_adavocab_pred = ada_head_pred_tokens_greedy[fail_tokens_pos]
        # TODO: Analyze the context of failed tokens
        adavocab_gt_fail_map = list(zip(fail_tokens_adavocab_gt.tolist(), fail_tokens_adavocab_pred.tolist()))

        origin_fail_tokens.extend(fail_tokens_origin.tolist())
        adavocab_fail_tokens.extend(fail_tokens_adavocab_gt.tolist())

        # Evaluate Label Rank
        origin_label_rank = get_label_rank_in_logits(origin_shift_lm_logits_flatten, shift_labels_flatten)
        origin_label_ranks.append(origin_label_rank)
        adavocab_label_rank = get_label_rank_in_logits(adavocab_shift_lm_logits_flatten, shift_labels_flatten)
        adavocab_label_ranks.append(adavocab_label_rank)

adavocab_fail_cases = set(adavocab_fail_tokens) - set(origin_fail_tokens)
adavocab_fail_cases_counter = {token: adavocab_fail_tokens.count(token) for token in adavocab_fail_cases}
print(detokenize(adavocab_fail_cases))

# Plot Label Rank
origin_label_ranks = torch.stack(origin_label_ranks)
adavocab_label_ranks = torch.stack(adavocab_label_ranks)

plot_label_ranking(origin_label_ranks.type(torch.float).cpu(), adavocab_label_ranks.type(torch.float).cpu(), mean=True)