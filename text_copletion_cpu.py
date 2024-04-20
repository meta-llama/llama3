from llama import Llama
from typing import List

import os
import torch
import torch.distributed as dist
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
dist.init_process_group(backend='gloo', init_method='env://')

ckpt_dir = "Meta-Llama-3-8B"
tokenizer_path = "Meta-Llama-3-8B/tokenizer.model"
temperature: float = 0.6
top_p: float = 0.9
max_seq_len: int = 128
max_gen_len: int = 64
max_batch_size: int = 4

generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

prompts: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
  
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to Ukrainian:

        sea otter => морська видра
        peppermint => перцева м‘ята
        plush girafe => плюшева жирафа
        cheese => """,
    ]

results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

for prompt, result in zip(prompts, results):
    print(prompt)
    print(f"> {result['generation']}")
    print("\n==================================\n")