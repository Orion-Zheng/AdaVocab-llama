import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import bitsandbytes as bnb
from transformers import AutoConfig, AutoModelForCausalLM
import copy
import bitsandbytes as bnb
# Loosely based on:  https://github.com/huggingface/transformers/issues/31474
@torch.no_grad()
def quantize_bnb_4(weight):
    out_features, in_features = weight.shape
    w = bnb.nn.LinearNF4(
        in_features,
        out_features,
        bias=False,
    )
    w.weight = bnb.nn.Params4bit(
        weight, requires_grad=False, quant_type="nf4", blocksize=64,
    ).to(weight.dtype)
    return w

@torch.no_grad()
def quantize_bnb_8(weight):
    out_features, in_features = weight.shape
    w = bnb.nn.Linear8bitLt(
        in_features,
        out_features,
        bias=False,
        has_fp16_weights=False,
        threshold=6.0,  # Default from the LLM.int8() paper
    )
    w.weight = bnb.nn.Int8Params(
        weight, requires_grad=False, has_fp16_weights=False
    ).to(weight.dtype)
    
    return w

def quant_lm_head(model, quant_type='int8'):
    if quant_type == 'int8':
        quantize = quantize_bnb_8
    elif quant_type == 'nf4':
        quantize = quantize_bnb_4
    else:
        raise ValueError(f"Unsupported quant_type: {quant_type}")
    model._modules['lm_head'] = quantize(model.lm_head.weight)
    model._modules['lm_head'].requires_grad_(False)
    return model

def fake_quant_lm_head(model, quant_type='int8'):
    if quant_type == 'int8':
        quantize = quantize_bnb_8
    elif quant_type == 'nf4':
        quantize = quantize_bnb_4
    else:
        raise ValueError(f"Unsupported quant_type: {quant_type}")
    out_features, in_features = model.lm_head.weight.shape
    quantized_linear = quantize(model.lm_head.weight).cuda()
    model.lm_head.weight.data = quantized_linear(torch.eye(in_features).half().cuda()).T.contiguous().cpu()
    return model

def test_model(model, tokenizer, input):
    inputs = tokenizer(input, add_special_tokens=False, return_tensors='pt')
    print('INPUT: ', tokenizer.decode(inputs['input_ids'][0]))
    inputs = inputs.to(model.device)
    pred = model.generate(**inputs,
                        max_new_tokens=32,
                        do_sample=False,
                        num_return_sequences=1)
    result = tokenizer.decode(pred.cpu()[0])
    del model 
    torch.cuda.empty_cache()
    return result

def add_chat_template(test_input, tokenizer):
    chat = [
        { "role": "user", "content": test_input },
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return prompt

if __name__ == "__main__":
    test_input = """Q: Who won Super Bowl XX?\n\nA:"""
    test_input = """What should I do to be a good researcher?"""

    model_id = "experiment_ckpts/gemma-2b_SFT-2024-06-10-123619/checkpoint-11592"  # WildChat-SFT-ep2
    model_id = "experiment_ckpts/qwen2-1.5b_SFT-2024-06-10-125011/checkpoint-12212"  # WildChat-SFT-ep2
    # model_id = "google/gemma-2b"
    # model_id = "google/gemma-7b"
    # model_id = "Qwen/Qwen2-1.5B"
    # model_id = "Qwen/Qwen2-7B"
    # model_id = "meta-llama/Meta-Llama-3-8B"
    # model_id = "TinyLlama/TinyLlama_v1.1"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # test_input = add_chat_template(test_input, tokenizer)
    print('Testing model:', model_id)   
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='cpu')
    print('===== Original Model Output ===== \n', test_model(model.cuda(), tokenizer, test_input))
    
    fake_model_8bit = fake_quant_lm_head(copy.deepcopy(model), quant_type='int8')
    # print(torch.allclose(model.lm_head.weight.data, fake_model_8bit.lm_head.weight.data))
    print('===== Int-8 Model Output ===== \n', test_model(fake_model_8bit.cuda(), tokenizer, test_input))
    del fake_model_8bit
    fake_model_4bit = fake_quant_lm_head(copy.deepcopy(model), quant_type='nf4')
    # print(torch.allclose(model.lm_head.weight.data, fake_model_4bit.lm_head.weight.data))
    print('===== NF-4 Model Output ===== \n', test_model(fake_model_4bit.cuda(), tokenizer, test_input))
    del fake_model_4bit