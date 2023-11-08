import torch
from torch import nn
from torch.nn import functional as F
from src import util
import numpy as np

class Model(nn.Module):
    # TODO: Parameterize device
    def __init__(self, vocab_sz, context_sz, embd_dim, num_heads, num_transformer_blocks, head_sz, lr):
        super().__init__()

        # Store params for saving and loading.
        self.init_params = {
            "vocab_sz": vocab_sz,
            "context_sz": context_sz,
            "embd_dim": embd_dim,
            "num_heads": num_heads,
            "num_transformer_blocks": num_transformer_blocks,
            "head_sz": head_sz,
            "lr": lr
        }

        # Token embedding layer
        self.tok_embd = nn.Embedding(vocab_sz, embd_dim)

        # Positional encodings
        self.pos_embd = nn.Embedding(context_sz, embd_dim)

        # Transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embd_dim, num_heads, head_sz, context_sz) for _ in range(num_transformer_blocks)],
            nn.LayerNorm(embd_dim)
        )

        # Final language modelling head
        self.lm_head = nn.Linear(embd_dim, vocab_sz)

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

    def forward(self, x):
        # x: B, T (integer encoded)
        tok_embeddings = self.tok_embd(x)
        pos_embeddings = self.pos_embd(x)
        x = self.transformer_blocks(tok_embeddings + pos_embeddings) # (B, T, C)
        logits = self.lm_head(x) # vocab_sz
        return logits

    def loss(self, logits, y):
        """ Computes loss from logits as returned by forward() with respect to
        labels y. """
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.reshape(B*T, C), y.reshape(B*T))
        return loss

    def train_batch(self, x, y):
        """ Performes a single training step on batch data x, y """
        self.train()

        # Forward pass.
        logits = self(x)
        loss = self.loss(logits, y)

        # Backward pass.
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Parameter update.
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def eval_loss(self, x, y, chunk_sz=128):
        """ Computes loss in evaluation mode. """
        # ???: Why do we oom if we don't chunk the data set?
        training = self.training
        self.eval()
        losses = []
        for c in range(int(np.ceil(len(x) / chunk_sz))):
            x_chunk = x[c*chunk_sz: (c+1)*chunk_sz]
            y_chunk = y[c*chunk_sz: (c+1)*chunk_sz]
            logits = self.forward(x_chunk)
            loss = self.loss(logits, y_chunk)
            losses.append(loss.item())
        self.train(training)
        return np.mean(losses), np.std(losses)

    def save(self, path):
        # TODO: implement sane saving method
        pass

    def load(self, path):
        # TODO
        pass

    @torch.no_grad()
    def generate(self, x: torch.Tensor, num_tokens=1):
        # Be friendly towards 1D inputs without batch dimension. We add a flat
        # dimension so the model get the shape of data it expects.
        squeeze = False
        if x.dim() == 1:
            squeeze = True
            x = x.unsqueeze(0)

        training = self.training
        self.eval()

        # x is (B, T) tensor of indices.
        result = torch.empty((x.shape[0], num_tokens), dtype=x.dtype)
        for i in range(num_tokens):
            # Get logits for last time step
            logits = self(x)
            logits = logits[:, -1, :] # (B, vocab_sz)
            probs = logits.softmax(-1) # (B, vocab_sz)
            prediction = torch.multinomial(probs, num_samples=1) # (B, 1)
            result[:, i:i+1] = prediction
            x = torch.cat((x, prediction), dim=-1) # (B, T+1)

        self.train(training)

        # Squeeze flat dimension added earlier. Note that we can't just check
        # result.shape[0] == 1 and omit the squeeze state altogether, as we'd
        # then squeeze out batch dimensions of size 1, even if they were
        # provided by the caller.
        if squeeze:
            result = result.squeeze(0)

        return result

class AttentionHead(nn.Module):
    """ Single head of masked self-attention """
    def __init__(self, fan_in, head_sz, context_sz):
        super().__init__()
        # Save head size in order to apply scaling later.
        self.head_sz = head_sz
        self.query_mat = nn.Linear(fan_in, head_sz, bias=False)
        self.key_mat = nn.Linear(fan_in, head_sz, bias=False)
        self.value_mat = nn.Linear(fan_in, head_sz, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones((context_sz, context_sz))))


    def forward(self, x):
        _, T, _ = x.shape

        query = self.query_mat(x)
        key = self.key_mat(x)
        value = self.value_mat(x)

        weights = query @ key.transpose(-2, -1)
        weights = weights / self.head_sz **0.5 # scale
        weights = weights.masked_fill(self.mask[:T, :T] == 0, float("-inf")) # mask
        weights = weights.softmax(dim=-1)

        # TODO: Add dropout later

        return weights @ value # B, T, head_sz


class MultiHeadAttention(nn.Module):
    """ Multi-head scaled self-attention with subsequent concatenation and linear layer """
    def __init__(self, in_features, num_heads, head_sz, context_sz, out_features=None):
        super().__init__()

        if out_features is None:
            out_features = in_features

        # TODO: Apparently, this list-based implementation is inefficient.
        #       Check Karpathy's nano-gpt for a more streamlined tensor implementation.
        self.heads = nn.ModuleList([AttentionHead(in_features, head_sz, context_sz) for _ in range(num_heads)])
        self.lin = nn.Linear(head_sz * num_heads, out_features)

    def forward(self, x):
        y = torch.cat([h(x) for h in self.heads], dim=-1)
        y = self.lin(y)
        return y


class FeedForward(nn.Module):
    """ Feedfoward network following a multi-head attention block """
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()

        if hidden_features is None:
            hidden_features = 4 * in_features
        if out_features is None:
            out_features = in_features

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class TransformerBlock(nn.Module):
    """ Transformer Block with residual connections and layer normalization """
    def __init__(self, in_features, num_head, head_sz, context_sz, hidden_features=None, out_features=None):
        super().__init__()

        if hidden_features is None:
            hidden_features = 4 * in_features
        if out_features is None:
            out_features = in_features

        self.layer_norm1 = nn.LayerNorm(in_features)
        self.self_attention = MultiHeadAttention(in_features, num_head, head_sz, context_sz, out_features=out_features)
        self.layer_norm2 = nn.LayerNorm(in_features)
        self.feed_forward = FeedForward(in_features, hidden_features=hidden_features, out_features=out_features)

    def forward(self, x):
        x = self.layer_norm1(x)
        x = x + self.self_attention(x)
        x = self.layer_norm2(x)
        x = x + self.feed_forward(x)
        return x

def get_batch(x, y, batch_sz):
    ix = torch.randint(len(x), (batch_sz,))
    return x[ix], y[ix]

# TODO: Write function to store files to uint8-encoded binary.
# TODO: Add padding character
# FIXME: Can't use uint8 as index with nn.Embedding. Is there an option to use uint8 indices?
def load_data(file, context_size, special_token="%", max_lines=1e12, dtype=torch.long):
    """ Tokenizes on a character level data from a file of newline-separated
    sequences of ascii text into a torch.tensor, one sample per line. The
    special token (index 0) is used to pad until max_len is reached. Lines
    longer than max_len are skipped altogether. """

    # First pass: Build up vocabulary and count lines to allocate memory later.
    chars = set()
    num_lines = 0 # number of lines read
    for line in util.read_lines(file, max_len=context_size+1, max_lines=max_lines):
        num_lines += 1
        chars = chars | set(line)
    if special_token in chars:
        raise "Special token contained by input vocabulary"

    # Construct vocabulary, encoder and decoder.
    chars = [special_token] + sorted(list(chars)) # special token needs to have 0th index
    atoi = {c:i for i, c in enumerate(chars)}
    itoa = {i:c for c, i in atoi.items()}
    encode = lambda in_string: torch.as_tensor([atoi[c] for c in in_string], dtype=dtype) # takes string, returns tensor
    decode = lambda in_tensor: "".join([itoa[i.item()] for i in in_tensor]) # takes 1D tensor, returns string

    # Second pass: Fill preallocated zero tensor. Zero encodes the special
    # token here, hence we don't need to manually pad the sequences.
    data = torch.zeros((num_lines, context_size), dtype=dtype)
    for i, line in enumerate(util.read_lines(file, max_len=context_size+1, max_lines=max_lines)):
        tokens = encode(line)
        data[i, :len(tokens)] = tokens

        # Logging
        if i % max(1, (num_lines // 1000)) == 0 or i == num_lines-1:
            print("\r" * 100 + f"processed {i+1}/{num_lines} lines ... ", end="")

    return data, chars, encode, decode