import torch
import warnings
warnings.filterwarnings("ignore")
from hqq.core.quantize import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()

if __name__ == "__main__":
    chat_template = False
    device = torch.device('cuda')
    # device = torch.device('cpu')
    dtype = torch.float16
    # dtype = torch.float32
    nbits = 2

    model_path = "google/gemma-2b"
    model_path = "google/gemma-2b-it"
    # model_path = "google/gemma-2-2b"
    # model_path = "google/gemma-2-2b-it"
    
    quant_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    #Quantization settings
    quant_config = BaseQuantizeConfig(nbits=nbits, group_size=64)
    #Replace your linear layer 
    quant_lm_head = HQQLinear(quant_model.lm_head, #torch.nn.Linear or None 
                              quant_config=quant_config, #quantization configuration
                              compute_dtype=dtype, #compute dtype
                              device=quant_model.device, #cuda device
                              initialize=True, #Use False to quantize later
                              del_orig=True #if True, delete the original layer
                             )
    quant_model.lm_head.weight.data = quant_lm_head.dequantize()  # Fake Quantization
    quant_model.eval()
    quant_model.save_pretrained('experiment_models/'+model_path+f'_hqq_{nbits}bit_head')