from __future__ import annotations

import time
import subprocess
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F

from .base_transformer import BaseTransformer


class VanillaTransformer(nn.Module, BaseTransformer):
    # TODO: head_size aus anderen Parametern automatisch bestimmen
    # TODO: ChessTransformer Klasse verwenden, die vocab_sz automatisch aus Vokabular bestimmt
    def __init__(
        self,
        vocab_sz,
        context_sz,
        embd_dim,
        num_heads,
        num_transformer_blocks,
        head_sz,
        lr,
    ):
        super().__init__()

        # Store params for saving and loading.
        self.init_params = {"vocab_sz": vocab_sz, "context_sz": context_sz, "embd_dim": embd_dim, "num_heads": num_heads, "num_transformer_blocks": num_transformer_blocks, "head_sz": head_sz, "lr": lr}

        # Token embedding layer
        self.tok_embd = nn.Embedding(vocab_sz, embd_dim)

        # Positional encodings
        self.pos_embd = nn.Embedding(context_sz, embd_dim)

        # Transformer blocks
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(embd_dim, num_heads, head_sz, context_sz) for _ in range(num_transformer_blocks)], nn.LayerNorm(embd_dim))

        # Final language modelling head
        self.lm_head = nn.Linear(embd_dim, vocab_sz)

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

    def forward(self, x):
        """
        Returns the logits of the forward pass of `x`.
        """

        # x: B, T (integer encoded)
        tok_embeddings = self.tok_embd(x.int())  # allow for uint8 inputs
        pos_embeddings = self.pos_embd(x.int())  # TODO: Remove uint8 inputs, use int or long everywhere
        x = self.transformer_blocks(tok_embeddings + pos_embeddings)  # (B, T, C)
        logits = self.lm_head(x)  # vocab_sz
        return logits

    def loss(self, logits, y):
        """
        Computes loss from logits as returned by forward() with respect to
        labels y.
        """

        B, T, C = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * T, C),
            y.reshape(B * T),
            # TODO: Konvergiert nicht, wenn padding tokens ignoriert werden. Wallom?
            # ignore_index=PADDING_TOKEN_ID,
        )
        return loss

    def train_epoch(
        self,
        data_loader: DataLoader,
        *,
        checkpoint_every=None,
        checkpt_dir="./models",
    ):
        """
        Trains on data provided by a data loader.
        """

        start = time.time()
        batch_losses = []

        def checkpoint(run_eval=False):
            nonlocal batch_losses
            nonlocal batch_idx
            loss_mean, loss_std = np.mean(batch_losses), np.std(batch_losses)
            now = str(datetime.now())
            print(f"\nBatch {batch_idx}: mean batch loss: {loss_mean} +- {loss_std}")
            checkpt_path = checkpt_dir + "/" + f"checkpt-{now}.pth"
            self.save(checkpt_path)

            # Trigger detached full eval
            if run_eval:
                print("Starting full eval of model checkpoint in the background ...")
                subprocess.run(
                    ["python", "main.py", "eval-perf", checkpt_path],
                    start_new_session=True,
                    check=False,
                )

            batch_losses = []

        for batch_idx, (x, y) in enumerate(data_loader, 1):
            loss = self.train_batch(x, y)

            bps = batch_idx / (time.time() - start)
            eta = (len(data_loader) - batch_idx) / bps
            print("\r" * 100 + f"batch {batch_idx}/{len(data_loader)}: batch loss {loss:.4f} ({bps:.2f} bps, ETA {eta:.2f}s)", end="", flush=True)
            batch_losses.append(loss)

            if checkpoint_every is not None and batch_idx % checkpoint_every == 0:
                checkpoint()
        checkpoint(run_eval=False)

    def train_batch(self, x: torch.Tensor, y: torch.Tensor):
        """
        Performes a single training step on batch data x, y.
        """

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

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path) -> VanillaTransformer:
        return torch.load(path)


class AttentionHead(nn.Module):
    """
    Single head of masked self-attention.
    """

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
        weights = weights / self.head_sz**0.5  # scale
        weights = weights.masked_fill(self.mask[:T, :T] == 0, float("-inf"))  # mask
        weights = weights.softmax(dim=-1)

        # TODO: Add dropout later

        return weights @ value  # B, T, head_sz


class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled self-attention with subsequent concatenation and linear layer.
    """

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
    """
    Feedfoward network following a multi-head attention block.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()

        if hidden_features is None:
            hidden_features = 4 * in_features
        if out_features is None:
            out_features = in_features

        self.net = nn.Sequential(nn.Linear(in_features, hidden_features), nn.ReLU(), nn.Linear(hidden_features, out_features))

    def forward(self, x):
        out = self.net(x)
        return out


class TransformerBlock(nn.Module):
    """
    Transformer Block with residual connections and layer normalization.
    """

    def __init__(
        self,
        in_features,
        num_head,
        head_sz,
        context_sz,
        hidden_features=None,
        out_features=None,
    ):
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
