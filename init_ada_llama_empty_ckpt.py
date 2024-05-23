import os
import json
import argparse

from transformers import AutoTokenizer, AutoConfig

from codebase.utils import set_model_config
from adavocab_llama.ada_vocab_llama import AdaVocabLlamaForCausalLM

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AdaVocabLlama Initialization')
    parser.add_argument('--model_path', type=str, default='original_models/tinyllama-chat', help='Name of the model')
    parser.add_argument('--output_dir', type=str, default='base_models/', help='Output directory')
    parser.add_argument('--ada_config_file', type=str, help='Path to the adavocab config file')
    args = parser.parse_args()

    model_path = args.model_path
    output_dir = args.output_dir
    config_file = args.ada_config_file

    model_name = os.path.basename(os.path.normpath(model_path))
    # Extract the file name from the path
    config_name = os.path.splitext(os.path.basename(config_file))[0]

    with open(config_file, 'r') as f:
        ada_config = json.load(f)

    config = AutoConfig.from_pretrained(model_path)
    set_model_config(config, ada_config)

    output_path = os.path.join(output_dir, f'ada-{model_name}-empty_{config_name}')
    AdaVocabLlamaForCausalLM.from_pretrained(model_path, config=config).save_pretrained(output_path)
    AutoTokenizer.from_pretrained(model_path).save_pretrained(output_path)
    config.save_pretrained(output_path)