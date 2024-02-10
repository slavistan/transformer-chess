from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F

from .base_transformer import BaseTransformer


@dataclass
class TransformerConfig:
    context_size: int  # = 1024
    vocab_size: int  # = 50304
    n_layer: int  # = 12
    n_head: int  # = 12
    n_embd: int  # = 768
    dropout: float  # = 0.0
    bias: bool  # = False      #faster model with better results if set to false, GPT2 had it set to true tho
    device: str
    lr: float


class GigachadTransformer(nn.Module, BaseTransformer):
    def __init__(self, *, config):
        super().__init__()
        self.config: TransformerConfig = config

        # context magic
        self.ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

        self.transformer = nn.ModuleDict(
            dict(
                tok_emb=nn.Embedding(config.vocab_size, config.n_embd),
                pos_emb=nn.Embedding(config.context_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.register_buffer("pos_index", torch.arange(config.context_size, dtype=torch.long))
        # weight tying. I have no idea what this does or why this is useful,
        # will have to investigate
        self.transformer.tok_emb.weight = self.lm_head.weight

        # init weights
        self.apply(self._init_weights)

        # Normalization as per GPT2 paper
        for p_name, p in self.named_parameters():
            if p_name.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        self.to(self.config.device)

    def forward(self, idx):
        _, T = idx.shape

        tok_emb = self.transformer.tok_emb(idx)
        pos_emb = self.transformer.pos_emb(self.pos_index[:T])

        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
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
        )
        return loss

    def train_epoch_batch_accumulation(
        self,
        data_loader: DataLoader,
        num_acc_batches: int,
    ):
        """
        Trains on data provided by a data loader.
        """

        start = time.time()
        batch_losses = []

        self.train()
        i = 0
        for batch_idx, (x, y) in enumerate(data_loader, 1):
            x = x.to(self.config.device)
            y = y.to(self.config.device)

            with self.ctx:
                logits = self(x)
                loss = self.loss(logits, y) / num_acc_batches

            # Backward pass.
            loss.backward()

            if i % num_acc_batches == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            i += 1

            # timing
            bps = batch_idx / (time.time() - start)
            eta = (len(data_loader) - batch_idx) / bps
            print("\r" * 100 + f"batch {batch_idx}/{len(data_loader)}: batch loss {loss.item():.4f} ({bps:.2f} bps, ETA {eta:.2f}s)", end="", flush=True)
            batch_losses.append(loss)

    def train_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        """
        Performes a single training step on batch data x, y.
        """

        self.train()
        x = x.to(self.config.device)  # TODO: entfernen, Caller soll auf richtiges Device achten
        y = y.to(self.config.device)

        # Forward pass.
        with self.ctx:
            logits = self(x)
            loss = self.loss(logits, y)

        # Backward pass.
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Parameter update.
        self.optimizer.step()

        return loss.item()

    def save(
        self,
        path: str,
    ):
        torch.save(self, path)

    @staticmethod
    def load(
        path: str,
    ) -> GigachadTransformer:
        return torch.load(path)

    def _init_weights(self, module):
        # TODO: ???: ist das manuelle Kaiming Initialisierung? Sollte mit Fanout skalieren.
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = FeedForward(config)

    def forward(self, X):
        X = X + self.attn(self.ln_1(X))
        X = X + self.mlp(self.ln_2(X))
        return X


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, X):
        X = self.c_fc(X)
        X = F.gelu(X, approximate="tanh")  # pylint: disable=not-callable
        X = self.c_proj(X)
        X = self.dropout(X)
        return X


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        assert config.n_embd % config.n_head == 0
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

    def forward(self, X):
        B, T, C = X.shape
        q, k, v = self.c_attn(X).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)  # pylint: disable=not-callable
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
