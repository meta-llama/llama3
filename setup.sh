mkdir checkpoints
mkdir checkpoints/rlhf_1_llama3_7B_pt2925k_annealed_sft_s5000_notools_mp4
manifold get llama3_partner_engineering_only/tree/rlhf_1_llama3_7B_pt2925k_annealed_sft_s5000_notools_mp4/consolidated.0.pth checkpoints/rlhf_1_llama3_7B_pt2925k_annealed_sft_s5000_notools_mp4/consolidated.0.pth
manifold get llama3_partner_engineering_only/tree/rlhf_1_llama3_7B_pt2925k_annealed_sft_s5000_notools_mp4/consolidated.1.pth checkpoints/rlhf_1_llama3_7B_pt2925k_annealed_sft_s5000_notools_mp4/consolidated.1.pth
manifold get llama3_partner_engineering_only/tree/rlhf_1_llama3_7B_pt2925k_annealed_sft_s5000_notools_mp4/consolidated.2.pth checkpoints/rlhf_1_llama3_7B_pt2925k_annealed_sft_s5000_notools_mp4/consolidated.2.pth
manifold get llama3_partner_engineering_only/tree/rlhf_1_llama3_7B_pt2925k_annealed_sft_s5000_notools_mp4/consolidated.3.pth checkpoints/rlhf_1_llama3_7B_pt2925k_annealed_sft_s5000_notools_mp4/consolidated.3.pth
manifold get llama3_partner_engineering_only/tree/v3_7b_2925k_annealed_mp1/cl_toplang_128k checkpoints/rlhf_1_llama3_7B_pt2925k_annealed_sft_s5000_notools_mp4/tokenizer.model
echo """{
    \"dim\": 4096,
    \"n_layers\": 32,
    \"n_heads\": 32,
    \"n_kv_heads\": 8,
    \"vocab_size\": 128256,
    \"multiple_of\": 1024,
    \"ffn_dim_multiplier\": 1.3,
    \"norm_eps\": 1e-05
}""" > checkpoints/rlhf_1_llama3_7B_pt2925k_annealed_sft_s5000_notools_mp4/params.json
