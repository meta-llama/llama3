#!/bin/bash
#torchrun --nproc_per_node 1 example_chat_completion.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 512 --max_batch_size 6
torchrun --nproc_per_node 1 example_chat_completion.py --ckpt_dir Meta-Llama-3-8B/ --tokenizer_path Meta-Llama-3-8B/tokenizer.model --max_seq_len 512 --max_batch_size 6
