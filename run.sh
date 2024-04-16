python example_chat_completion.py \
    --ckpt_dir checkpoints/rlhf_1_llama3_7B_pt2925k_annealed_sft_s5000_notools_mp4 \
    --tokenizer_path checkpoints/rlhf_1_llama3_7B_pt2925k_annealed_sft_s5000_notools_mp4/tokenizer.model \
    --model_name model.pth \
    --quantization_mode "no" \
    --torch_compile False \
    --max_seq_len 1024 --max_batch_size 1
