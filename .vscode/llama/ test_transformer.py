# Import necessary modules
import torch

class ModelArgs:
    def __init__(self, dim, n_layers, n_heads, vocab_size, max_batch_size, max_seq_len):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

# Initialize model parameters
params = ModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    vocab_size=10000,
    max_batch_size=32,
    max_seq_len=2048,
)

# Initialize the Transformer model (ensure this is defined in your code)
model = Transformer(params) # type: ignore

# Create dummy input data
tokens = torch.randint(0, params.vocab_size, (params.max_batch_size, params.max_seq_len))
start_pos = 0

# Pass the input data through the model
output = model(tokens, start_pos)
print(output.shape)
