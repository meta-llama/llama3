# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math  # Importing math for mathematical operations
from dataclasses import dataclass  # Importing dataclass for creating data classes
from typing import Optional, Tuple, List  # Importing type hints for better code clarity

# Importing necessary libraries from FairScale for model parallelism
import fairscale.nn.model_parallel.initialize as fs_init
import torch  # Importing PyTorch for tensor operations
import torch.nn.functional as F  # Importing functional API for neural network operations
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,  # Layer for column-parallel linear operations
    RowParallelLinear,     # Layer for row-parallel linear operations
    VocabParallelEmbedding, # Layer for vocabulary embedding with parallelism
)
from torch import nn  # Importing nn module for building neural networks

@dataclass
class ModelArgs:
    """
    Class to hold the arguments for the model configuration.
    """

    dim: int = 4096  # Dimension of the model's embeddings and hidden layers
    n_layers: int = 32  # Number of transformer layers in the model
    n_heads: int = 32  # Number of attention heads in each layer
    n_kv_heads: Optional[int] = None  # Number of key/value attention heads (optional)
    vocab_size: int = -1  # Size of the vocabulary for embedding
    multiple_of: int = 256  # Ensure that hidden layer sizes are a multiple of this value
    ffn_dim_multiplier: Optional[float] = None  # Multiplier for the feedforward layer's hidden dimension (optional)
    norm_eps: float = 1e-5  # Epsilon value for normalization layers to prevent division by zero
    rope_theta: float = 500000  # Parameter for rotary embeddings

    max_batch_size: int = 32  # Maximum number of samples in a batch
    max_seq_len: int = 2048  # Maximum sequence length for input data



class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize RMSNorm with the given dimension and epsilon.

        Args:
            dim (int): The dimensionality of the input.
            eps (float): A small constant to avoid division by zero.
        """
        super().__init__()  # Initialize the parent class
        self.eps = eps  # Set the epsilon value
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable weight parameter initialized to ones

    def _norm(self, x):
        """Compute the RMS normalization.

        Args:
            x (Tensor): The input tensor to be normalized.

        Returns:
            Tensor: Normalized tensor.
        """
        # Calculate the RMS normalization
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """Forward pass for RMSNorm.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Normalized output tensor with applied weights.
        """
        output = self._norm(x.float()).type_as(x)  # Normalize input and cast back to original type
        return output * self.weight  # Scale the normalized output by the learnable weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute frequency components for complex numbers used in position encoding.

    Args:
        dim (int): The dimensionality for the frequency components.
        end (int): The maximum time step or length for the frequency computation.
        theta (float): Scaling factor for the frequency calculation (default is 10000.0).

    Returns:
        Tensor: A tensor containing the complex exponential values for position encoding.
    """
    # Compute frequencies for the even dimensions
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    
    # Create a tensor representing the time steps
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    
    # Compute the outer product to generate the frequency matrix
    freqs = torch.outer(t, freqs)
    
    # Create complex numbers using the frequencies as angles
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_cis  # Return the computed complex frequency tensor


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """Reshape frequency tensor for broadcasting with another tensor.

    Args:
        freqs_cis (torch.Tensor): The tensor containing frequency components with shape (seq_len, head_dim).
        x (torch.Tensor): The tensor to which the frequency tensor will be broadcasted.

    Returns:
        torch.Tensor: A reshaped version of `freqs_cis` suitable for broadcasting with `x`.
    """
    # Get the number of dimensions of the input tensor x
    ndim = x.ndim
    
    # Ensure that the second dimension is valid (i.e., at least 2D)
    assert 0 <= 1 < ndim
    
    # Check that the shape of freqs_cis matches the required shape based on x
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    
    # Create a new shape for freqs_cis, where:
    # - Keep dimensions 1 (for broadcasting) or the last dimension unchanged
    # - Other dimensions are set to 1 for broadcasting
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    
    # Reshape freqs_cis to the new shape and return
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    Args:
        xq (torch.Tensor): The query tensor with shape (..., head_dim).
        xk (torch.Tensor): The key tensor with shape (..., head_dim).
        freqs_cis (torch.Tensor): The tensor containing precomputed frequency components for rotary embeddings.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The transformed query and key tensors after applying rotary embeddings.
    """
    # Convert query tensor to complex numbers by reshaping it into pairs of real components
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    
    # Convert key tensor to complex numbers in the same way
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Reshape freqs_cis to be compatible for broadcasting with the complex query tensor
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # Apply rotary embeddings to the query tensor and convert back to real numbers
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    
    # Apply rotary embeddings to the key tensor and convert back to real numbers
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    # Return the output tensors as the same type as the input tensors
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key-value tensor for each head.

    Args:
        x (torch.Tensor): The input tensor with shape (batch_size, sequence_length, n_kv_heads, head_dim).
        n_rep (int): The number of times to repeat the key-value heads.

    Returns:
        torch.Tensor: The tensor with repeated key-value heads, shape (batch_size, sequence_length, n_kv_heads * n_rep, head_dim).
    """
    # Get the shape of the input tensor
    bs, slen, n_kv_heads, head_dim = x.shape
    
    # If n_rep is 1, return the original tensor as no repetition is needed
    if n_rep == 1:
        return x
    
    # Expand the tensor to add a new dimension for repetition, then reshape it
    return (
        x[:, :, :, None, :]  # Add a new dimension for repetition
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # Repeat the tensor along the new dimension
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # Reshape to combine the repeated heads
    )

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        # Set up the number of key-value heads based on provided arguments
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        
        # Get the size of the model parallel world (number of GPUs)
        model_parallel_size = fs_init.get_model_parallel_world_size()
        
        # Calculate the number of local heads per model parallel unit
        self.n_local_heads = args.n_heads // model_parallel_size
        
        # Calculate the number of local key-value heads
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        
        # Determine the number of repetitions based on local heads and key-value heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        
        # Calculate the dimension of each attention head
        self.head_dim = args.dim // args.n_heads

        # Initialize weight matrices for queries, keys, and values
        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        
        # Initialize the output linear layer for combining attention outputs
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        # Initialize cache tensors for keys and values for efficient attention computation
        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()  # Allocate on GPU
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()  # Allocate on GPU

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        # Get batch size and sequence length from input tensor
        bsz, seqlen, _ = x.shape
        
        # Compute queries, keys, and values using weight matrices
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape the queries, keys, and values for attention heads
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # Apply rotary embeddings to queries and keys
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Store current keys and values in the cache for future use
        self.cache_k[:bsz, start_pos:start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos:start_pos + seqlen] = xv

        # Retrieve the relevant keys and values from the cache
        keys = self.cache_k[:bsz, :start_pos + seqlen]
        values = self.cache_v[:bsz, :start_pos + seqlen]

        # Repeat keys and values to match the number of query heads
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # Transpose to prepare for attention score calculation
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute attention scores using scaled dot-product attention
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Apply the mask to scores if provided
        if mask is not None:
            scores = scores + mask
            
        # Normalize scores using softmax
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # Compute the output by weighted sum of values based on attention scores
        output = torch.matmul(scores, values)
        
        # Reshape output back to the original format
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        # Apply the final linear transformation to the output
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        
        # Calculate the hidden dimension, adjusting based on the multiplier if provided
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)

        # Ensure the hidden dimension is a multiple of the specified value
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # Define the first linear transformation (input to hidden)
        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        
        # Define the second linear transformation (hidden to output)
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        
        # Define an additional linear transformation (input to hidden for residual connection)
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        # Apply the first linear transformation with activation,
        # then multiply by the output of the third linear transformation
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        
        # Store parameters from the model arguments
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        
        # Initialize the attention mechanism
        self.attention = Attention(args)
        
        # Initialize the feed-forward network with specified dimensions
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        
        # Store the layer identifier
        self.layer_id = layer_id
        
        # Initialize layer normalization for attention and feed-forward components
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        # Compute the attention output and add a residual connection
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        
        # Compute the feed-forward output and add another residual connection
        out = h + self.feed_forward(self.ffn_norm(h))
        
        # Return the final output of the transformer block
        return out

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        
        # Store model parameters
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # Initialize token embeddings
        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        # Create a list of transformer layers
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        # Initialize normalization layer for the final output
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        # Define the output linear layer to map to vocabulary size
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        # Precompute frequencies for rotary embeddings
        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        # Get the batch size and sequence length from tokens
        _bsz, seqlen = tokens.shape
        
        # Embed the input tokens
        h = self.tok_embeddings(tokens)
        
        # Move the frequency tensors to the same device as the input
        self.freqs_cis = self.freqs_cis.to(h.device)
        
        # Slice the frequency tensors for the current sequence
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        # Initialize mask for attention scores
        mask = None
        if seqlen > 1:
            # Create a mask for future tokens
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

            # Handle key-value caching by ensuring masked entries correspond to future tokens
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        # Pass the embeddings through each transformer layer
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        
        # Apply normalization to the final output
        h = self.norm(h)
        
        # Generate output logits by mapping to vocabulary size
        output = self.output(h).float()
        
        return output


