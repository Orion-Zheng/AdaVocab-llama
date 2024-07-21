import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.gemma import GemmaForCausalLM
import bitsandbytes as bnb
import copy
from hqq.core.quantize import *
from transformers import AutoModelForCausalLM, HqqConfig
from hqq.core.quantize import BaseQuantizeConfig as HQQBaseQuantizeConfig


def add_chat_template(test_input, tokenizer):
    chat = [
        { "role": "user", "content": test_input },
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return prompt

def infer_test(model, prompt, device, chat_template=False):
    if chat_template:
        prompt = add_chat_template(prompt, tokenizer)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
    else:
        inputs = tokenizer(prompt, add_special_tokens=True, return_tensors='pt')
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






model_id = "reflectio/adavocab-gemma-2b-512-offload"
chat_template = True
test_input = """What should I do to be a good researcher?"""

device = 'cuda'
compute_dtype = torch.float16
# Each linear layer with the same tag will use a dedicated quantization config
q4_config = {'nbits':2, 'group_size':64, 'quant_zero':False, 'quant_scale':False}
q3_config = {'nbits':2, 'group_size':32, 'quant_zero':False, 'quant_scale':False}
quant_config  = HqqConfig(dynamic_config={
#   'self_attn.q_proj':q4_config,
#   'self_attn.k_proj':q4_config,
#   'self_attn.v_proj':q4_config,
#   'self_attn.o_proj':q4_config,
#   'mlp.gate_proj':q3_config,
#   'mlp.up_proj'  :q3_config,
#   'mlp.down_proj':q3_config,
  'adavocab_head.A': q4_config,
  'adavocab_head.B': q4_config,
  
}) 
adavocab_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=compute_dtype, 
    # device_map=device, 
    quantization_config=quant_config,
    trust_remote_code=True,
)


tokenizer = AutoTokenizer.from_pretrained(model_id)
adavocab_model.eval()
adavocab_model.offload()

torch.cuda.reset_peak_memory_stats()
a_pred, a_elapsed_time = infer_test(adavocab_model, test_input, device, chat_template=chat_template)
print("======================================")
print(f"AdaVocab Model\n Elapsed time: {a_elapsed_time} seconds\n Maximum GPU memory usage: {torch.cuda.max_memory_allocated() / 1024**2} MB")
del adavocab_model
torch.cuda.empty_cache()

print("======================================")
print('===== Output from Quant AdaVocab Model =====')
print(f"{a_pred}")