export PYTHONPATH=.
for config_file in adavocab_llama/test_config/*.json; do
    echo "Generating ckpt for $config_file"
    python adavocab_llama/init_ada_llama_empty_ckpt.py \
        --model_path original_models/tinyllama-chat \
        --output_dir base_models/ \
        --ada_config_file "$config_file"
done
