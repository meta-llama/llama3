# Copyright (c) [Your Name or Company Name]
# This software may be used and distributed in accordance with the terms of the ©18BluntWrapz License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from bluntwrapz.model import BluntWrapzModelArgs, BluntWrapzTransformer
from bluntwrapz.tokenizer import BluntWrapzChatFormat, BluntWrapzDialog, BluntWrapzMessage, BluntWrapzTokenizer

class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required

class ChatPrediction(TypedDict, total=False):
    generation: BluntWrapzMessage
    tokens: List[str]  # not required
    logprobs: List[float]  # not required

class BluntWrapz:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "BluntWrapz":
        """
        Build a BluntWrapz instance by initializing and loading a model checkpoint.
        """
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"No checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(checkpoints), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: BluntWrapzModelArgs = BluntWrapzModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = BluntWrapzTokenizer(model_path=tokenizer_path)
        assert model_args.vocab_size == tokenizer.n_words
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = BluntWrapzTransformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return BluntWrapz(model, tokenizer)

    def __init__(self, model: BluntWrapzTransformer, tokenizer: BluntWrapzTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = BluntWrapzChatFormat(tokenizer)

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.
        """
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens))

        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                import torch
import torch.nn.functional as F
from typing import List, Optional

def sample_top_p(probs, top_p):
    """
    Samples from the top p portion of the probability distribution.
    probs: Tensor of token probabilities.
    top_p: The cumulative probability threshold.
    """
    # Sort the probabilities in descending order and their corresponding indices
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Compute the cumulative sum of probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Remove tokens with cumulative probability above the threshold (top_p)
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # Shift the indices to keep at least one token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Set probabilities of tokens that should be removed to 0
    sorted_probs[sorted_indices_to_remove] = 0.0

    # Normalize the remaining probabilities
    sorted_probs /= sorted_probs.sum()

    # Sample from the remaining tokens
    next_token = torch.multinomial(sorted_probs, 1)

    return sorted_indices[next_token]


class LanguageModel:
    def __init__(self, model, tokenizer, formatter):
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = formatter

    def generate(
        self, prompt_tokens, max_gen_len, temperature, top_p, logprobs, echo
    ):
        min_prompt_len = min([len(toks) for toks in prompt_tokens])
        total_len = max_gen_len + min_prompt_len

        tokens = torch.zeros((len(prompt_tokens), total_len), dtype=torch.long)
        input_text_mask = torch.zeros_like(tokens, dtype=torch.bool)
        eos_reached = torch.zeros((len(prompt_tokens),), dtype=torch.bool)

        for i, toks in enumerate(prompt_tokens):
            tokens[i, : len(toks)] = torch.tensor(toks)
            input_text_mask[i, : len(toks)] = 1

        prev_pos = 0
        token_logprobs = (
            torch.zeros((len(prompt_tokens), total_len), dtype=torch.float)
            if logprobs
            else None
        )
        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens))

        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token

            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=self.tokenizer.pad_token_id,
                )

            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()

        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][
                    start : len(prompt_tokens[i]) + max_gen_len
                ]
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                    probs = probs[:eos_idx] if logprobs else None
                except ValueError:
                    pass
            

