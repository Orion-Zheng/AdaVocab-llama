import json
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from adavocab_llama.ada_vocab_factory import AdaVocabLlamaForCausalLM, AdaVocabGemmaForCausalLM, AdaVocabQwen2ForCausalLM
import matplotlib.pyplot as plt

def load_mt_bench_answers(file_path):
    dataset = []
    with open(file_path, "r") as file:
        for line in file:
            json_obj = json.loads(line)
            item = {'input_ids': torch.tensor(json_obj['total_token_ids']), 'attention_mask': torch.tensor(json_obj['attention_mask'])}
            dataset.append(item)
    return dataset

def detokenize(tokens):
    return tokenizer.convert_ids_to_tokens(list(tokens))

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
    top_k_indices = torch.topk(lm_head_logits, k, dim=-1).indices  # (bs*seq_len - 1, k)
    label_ids_in_topk = (top_k_indices == labels.unsqueeze(1)).any(-1)  # (bs*seq_len - 1, )
    label_not_ignore = ~(labels == IGNORE_INDEX)
    return labels[~label_ids_in_topk & label_not_ignore]

def labels_not_select_by_adavocab(ada_logits, labels, IGNORE_INDEX=-100):
    # shifted_ada_logits: (bs*seq_len - 1, vocab_size)
    # labels: (seq_len - 1, )
    ids_not_ignore = ~(labels == IGNORE_INDEX)
    trunc_labels = labels[ids_not_ignore]
    trunc_ada_logits = ada_logits[ids_not_ignore]
    trunc_adavocab_selected = (trunc_ada_logits > 0)
    trunc_label_selected_mask = trunc_adavocab_selected.gather(-1, trunc_labels.unsqueeze(1)).flatten()
    trunc_label_not_selected_mask = ~trunc_label_selected_mask
    
    labels_idx_miss_by_adavocab = torch.zeros_like(labels, dtype=torch.bool)
    labels_idx_miss_by_adavocab[ids_not_ignore] = trunc_label_not_selected_mask

    return labels[labels_idx_miss_by_adavocab], labels_idx_miss_by_adavocab

device = "cuda"
model_path = "google/gemma-1.1-2b-it"
adavocab_model_path = "/net/papilio/storage7/tingyuan/llama/adavocab_eval/ckpts/Ada_Gemma-2b-FT-2024-07-10-010024-512/checkpoint-2515"
dataset_path = "adavocab_llama/token_level_analysis/google/gemma-1.1-2b-it_mt_answers.jsonl"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
# adavocab_model = AdaVocabQwen2ForCausalLM.from_pretrained(adavocab_model_path, torch_dtype=torch.float16, device_map=device)
adavocab_model = AdaVocabGemmaForCausalLM.from_pretrained(adavocab_model_path, torch_dtype=torch.float16, device_map=device)

tokenized_dataset = load_mt_bench_answers(dataset_path)

IGNORE_INDEX = -100
max_length = 2048 
ignore_begin_tokens = 100  # ignore the loss on first 100 tokens
adavocab_model.eval()
model.eval()
nlls = []

origin_fail_tokens = []
adavocab_fail_tokens = []

origin_label_ranks = []
adavocab_label_ranks = []

fail_fragments_pairs = []
for example in tqdm(tokenized_dataset):
    input_ids = example['input_ids'].to(device)
    target_ids = example['input_ids'].to(device) 
    target_ids[example['attention_mask']==0] = -100
    input_ids = input_ids[None, ...]
    target_ids = target_ids[None, ...]
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
        
        # ce_loss = nn.CrossEntropyLoss(reduction="none")
        # origin_loss = ce_loss(origin_shift_lm_logits_flatten, shift_labels_flatten)  # position-wise loss
        # adavocab_loss = ce_loss(adavocab_shift_lm_logits_flatten, shift_labels_flatten)  # position-wise loss
        
        fail_tokens_origin = labels_not_in_top_k_logits(origin_shift_lm_logits_flatten, shift_labels_flatten, adavocab_model.config.ADA_TOPK)
        fail_tokens_adavocab_gt, fail_tokens_pos = labels_not_select_by_adavocab(ada_head_logits_flatten, shift_labels_flatten)
        ada_head_pred_tokens_greedy = adavocab_shift_lm_logits_flatten.argmax(-1)
        fail_tokens_adavocab_pred = ada_head_pred_tokens_greedy[fail_tokens_pos]
        origin_head_pred_tokens_greedy = origin_shift_lm_logits_flatten.argmax(-1)

        for pos in torch.nonzero(fail_tokens_pos).squeeze(1).tolist():  # one sequence at a time
            gt_context = shift_labels_flatten[pos-10:pos]
            gt_context = torch.where(gt_context == -100, torch.tensor(0), gt_context)
            gt_label = shift_labels_flatten[pos:pos+1]
            gt_label = torch.where(gt_label == -100, torch.tensor(0), gt_label)
            origin_pred = origin_head_pred_tokens_greedy[pos:pos+1]
            origin_pred = torch.where(origin_pred == -100, torch.tensor(0), origin_pred)
            ada_pred = ada_head_pred_tokens_greedy[pos:pos+1]
            ada_pred = torch.where(ada_pred == -100, torch.tensor(0), ada_pred)
            fail_fragments_pairs.append((gt_context, gt_label, origin_pred, ada_pred))
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
sorted_adavocab_fail_cases_counter = dict(sorted(adavocab_fail_cases_counter.items(), key=lambda item: item[1], reverse=True))
# print(detokenize(sorted_adavocab_fail_cases_counter))
for fail_pairs in fail_fragments_pairs:
    print("======================================")
    print('Context:', tokenizer.decode(fail_pairs[0]))
    # print('Ground Truth:\t', tokenizer.decode(fail_pairs[1]))
    print('Origin:\t\t', tokenizer.decode(fail_pairs[2]))
    print('AdaVocab:\t', tokenizer.decode(fail_pairs[3]))

# Plot Label Rank
# origin_label_ranks = torch.stack(origin_label_ranks)
# adavocab_label_ranks = torch.stack(adavocab_label_ranks)

# plot_label_ranking(origin_label_ranks.type(torch.float).cpu(), adavocab_label_ranks.type(torch.float).cpu(), mean=True)