from __future__ import annotations

from enum import Enum, auto
import sys
import subprocess
from datetime import datetime
import math
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F
import chess

from src.tools import count_lines_in_file, torch_elem_size
from src.tan_chess import (
    TANMoveList,
    TANMove,
    TANPlayer,
    TAN_MOVELINE_CHARS,
    TAN_MAX_MOVE_LEN,
    is_valid_move,
    tan_moveline_from_gameline,
)


class Model(nn.Module):
    def __init__(
        self,
        vocab_sz,
        context_sz,
        embd_dim,
        num_heads,
        num_transformer_blocks,
        head_sz,
        lr,
        *,
        device=None,
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

        if device is not None:
            self.to(device)

    def forward(self, x):
        # x: B, T (integer encoded)
        tok_embeddings = self.tok_embd(x.int())  # allow for uint8 inputs
        pos_embeddings = self.pos_embd(x.int())
        x = self.transformer_blocks(tok_embeddings + pos_embeddings)  # (B, T, C)
        logits = self.lm_head(x)  # vocab_sz
        return logits

    def loss(self, logits, y):
        """
        Computes loss from logits as returned by forward() with respect to
        labels y.
        """

        B, T, C = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, C), y.reshape(B * T))
        return loss

    def train_epoch(self, data_loader, *, checkpoint_every=None, checkpt_dir="./models"):
        """
        Trains on data provided by a data loader.
        """

        batch_losses = []

        def checkpoint(run_eval=False):
            nonlocal batch_losses
            nonlocal batch_idx
            loss_mean, loss_std = np.mean(batch_losses), np.std(batch_losses)
            now = str(datetime.now())
            print(f"Batch {batch_idx}: mean batch loss: {loss_mean} +- {loss_std}")
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

        for batch_idx, (x, y) in enumerate(data_loader, 0):
            loss = self.train_batch(x, y)
            batch_losses.append(loss)

            if checkpoint_every is not None and batch_idx % checkpoint_every == 0:
                checkpoint()
        checkpoint(run_eval=True)

    def train_batch(self, x, y):
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

    @torch.no_grad()
    def eval_loss(self, x, y, chunk_sz=128):
        """
        Computes loss in evaluation mode.
        """

        # ???: Why do we oom if we don't chunk the data set?
        training = self.training
        self.eval()
        losses = []
        num_chunks = int(np.ceil(len(x) / chunk_sz))
        for c in range(num_chunks):
            x_chunk = x[c * chunk_sz : (c + 1) * chunk_sz]
            y_chunk = y[c * chunk_sz : (c + 1) * chunk_sz]
            logits = self.forward(x_chunk)
            loss = self.loss(logits, y_chunk)
            losses.append(loss.item())
        self.train(training)
        return np.mean(losses), np.std(losses)

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path) -> Model:
        return torch.load(path)

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
            logits = logits[:, -1, :]  # (B, vocab_sz)
            probs = logits.softmax(-1)  # (B, vocab_sz)
            prediction = torch.multinomial(probs, num_samples=1)  # (B, 1)
            result[:, i : i + 1] = prediction
            x = torch.cat((x, prediction), dim=-1)  # (B, T+1)

        self.train(training)

        # Squeeze flat dimension added earlier. Note that we can't just check
        # result.shape[0] == 1 and omit the squeeze state altogether, as we'd
        # then squeeze out batch dimensions of size 1, even if they were
        # provided by the caller.
        if squeeze:
            result = result.squeeze(0)

        return result


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


class TransformerPlayer(TANPlayer):
    board: chess.Board
    model: Model
    model_context_sz: int
    model_device: str
    movetensor: torch.Tensor
    write_idx: int
    context_overflow: bool
    num_tries_until_valid: int

    def __init__(
        self,
        model: Model,
        movelist=(),
        num_tries_until_valid=1,
    ):
        self.model = model
        self.model_context_sz = self.model.init_params["context_sz"]
        self.model_device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.reset(movelist, num_tries_until_valid)

    def push_moves(
        self,
        movelist: TANMoveList,
    ) -> TransformerPlayer:
        if self.context_overflow:
            return self

        for m in movelist:
            encd = encode_moveline_as_np8(m + " ")
            num_tokens = len(encd)

            # Silently stop adding moves once we've reached the end of the
            # tensor. Resignation will follow.
            if self.write_idx + num_tokens >= len(self.movetensor):
                self.context_overflow = True
                return self

            self.movetensor[self.write_idx : self.write_idx + num_tokens] = torch.from_numpy(encd)
            self.write_idx += num_tokens
            self.board.push_san(m)

        return self

    def suggest_move(
        self,
    ) -> TransformerPlayer.ResignationReason | TANMove:
        if self.context_overflow:
            return TransformerPlayer.ResignationReason.CONTEXT_OVERFLOW

        for _ in range(self.num_tries_until_valid):
            movetensor_buffer = self.movetensor.clone()
            write_idx_buffer = self.write_idx

            while True:
                token = self.model.generate(movetensor_buffer[:write_idx_buffer], num_tokens=1).item()

                # Abort once we exceed the maximum number of characters a move
                # in TAN format can consist of.
                num_generated_tokens = write_idx_buffer - self.write_idx
                if num_generated_tokens >= TAN_MAX_MOVE_LEN:
                    break

                # Skip over padding tokens.
                # TODO: Padding tokens erscheinen in den Daten nach dem letzten
                #       Zug eines Spiels, sind also Teil eines sinnvollen Outputs.
                #       Vielleicht sollte ich padding tokens und whitespaces gleich
                #       behandeln.
                #       Alternativvorschlag: Spiele werden direkt mit einem extra Leerzeichen
                #                            am Ende der Zugsequenz kodiert, sodass Paddingtokens
                #                            niemals nach einem Zug erscheinen.
                if token == PADDING_IDX:
                    continue

                # Break if whitespace is returned. Whitespace is not part of
                # the returned move. However, continue if no tokens have been
                # generated yet.
                if token == WHITESPACE_IDX:
                    if num_generated_tokens == 0:
                        continue
                    break

                movetensor_buffer[write_idx_buffer] = token
                write_idx_buffer += 1

            # Decode generated move.
            move = decode_moveline_tensor(movetensor_buffer[self.write_idx : write_idx_buffer])
            if is_valid_move(move, self.board):
                return move

        return TransformerPlayer.ResignationReason.CANT_CONSTRUCT_VALID_MOVE

    def reset(
        self,
        movelist: TANMoveList = (),
        num_tries_until_valid=1,
    ) -> TransformerPlayer:
        self.board = chess.Board()
        self.movetensor = torch.zeros(
            (self.model_context_sz,),
            dtype=torch.uint8,
            device=self.model_device,
            requires_grad=False,
        )
        self.write_idx = 1  # write pointer; 0 is reserved for padding
        self.context_overflow = False
        self.num_tries_until_valid = num_tries_until_valid

        self.push_moves(movelist)

        return self

    class ResignationReason(Enum):
        CONTEXT_OVERFLOW = auto()
        """Preset context size of transformer exceeded by game tokens"""

        CANT_CONSTRUCT_VALID_MOVE = auto()
        """Failed to produce a valid move after a set amount of attempts"""


encode_moveline_dict = {c: np.uint8(i) for i, c in enumerate(TAN_MOVELINE_CHARS, 1)}
decode_moveline_dict = {i: c for c, i in encode_moveline_dict.items()}
WHITESPACE_IDX = encode_moveline_dict[" "]
PADDING_IDX = 0


def encode_moveline_as_np8(tan_moveline: str) -> np.ndarray:
    result = np.empty(len(tan_moveline), dtype=np.uint8)
    for i, c in enumerate(tan_moveline):
        result[i] = encode_moveline_dict[c]
    return result


def decode_moveline_tensor(tan_tokens: torch.Tensor) -> str:
    # TODO: Convert
    return "".join([decode_moveline_dict[np.uint8(t)] for t in tan_tokens.cpu()])


def decode_moveline(tan_tokens: Sequence[int]) -> str:
    return "".join([decode_moveline_dict[np.uint8(t)] for t in tan_tokens])


class Dump(Dataset):
    def __init__(self, tan_file: str, context_size: int, *, device=None, max_size=sys.maxsize, dtype=torch.uint8):
        width = context_size + 1  # tensor width
        max_lines = math.floor(max_size / torch_elem_size(dtype) / width)
        height = count_lines_in_file(tan_file, max_lines=max_lines)  # tensor height
        self.data = torch.zeros((height, width), dtype=dtype)
        with open(tan_file, "r") as f:
            for i, gameline in enumerate(f):
                if i >= max_lines:
                    break

                moveline = tan_moveline_from_gameline(gameline)
                n = min(len(moveline), context_size)
                encd = encode_moveline_as_np8(moveline[:n])
                self.data[i, 1 : n + 1] = torch.from_numpy(encd)

        if device is not None:
            self.data = self.data.to(device)

    def __getitem__(self, idx):
        return self.data[idx, :-1], self.data[idx, 1:]

    def __len__(self):
        return len(self.data)

    def mem_size(self):
        return self.data.nelement() * self.data.element_size()
