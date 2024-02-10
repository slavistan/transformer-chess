from __future__ import annotations
from abc import abstractmethod

import torch
from torch.utils.data import DataLoader


class BaseTransformer:
    @abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns 3D logits given 2D inputs x.
        """

    @abstractmethod
    def train_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> float:
        """
        Performs a single training step on batched inputs 'x' and labels 'y',
        returning the loss value for the batch.
        """

    @abstractmethod
    def save(
        self,
        path: str,
    ):
        """
        Saves the model to file.
        """

    @staticmethod
    @abstractmethod
    def load(
        path: str,
    ) -> BaseTransformer:
        """
        Loads a model from a file created by save().
        """

    @torch.no_grad()
    def prob_of_continuation(
        self,
        prefix: torch.Tensor,
        continuations: torch.Tensor,
        padding = -1,
    ) -> torch.Tensor:
        """
        Returns the conditional probability of 'continuations' given 'prefix'. Allows
        for batched 'continuations'. If continuations of different lengths are to be
        evaluated, pad the end using 'padding' as index.
        """
        assert len(continuations.shape) == 2, "expected a 2d tensor, one row per continuation"

        # TODO: optimieren, aufräumen und dokumentieren
        prefix = prefix.repeat(continuations.shape[0], 1)
        sequences = torch.cat((prefix, continuations), dim=1)

        mask = sequences == padding
        sequences[mask] = 0

        logits = self(sequences)
        probs = logits.softmax(-1)
        prob_mask = torch.cat((mask, torch.ones((sequences.shape[0], 1)).type(torch.bool)), dim=1)[:, 1:]
        probs[prob_mask, :] = 1.0

        t_idx = prefix.shape[1] - 1 + torch.arange(continuations.shape[1])
        probs_foo = torch.empty(continuations.shape, dtype=torch.float)
        # TODO: get rid of the loop
        for i in range(continuations.shape[0]):
            c_idx = sequences[i, prefix.shape[1] :]
            probs_cont = probs[i, t_idx, c_idx]
            probs_foo[i] = probs_cont
        probs_final = probs_foo.prod(dim=-1)

        assert len(probs_final.shape) == 1 and len(probs_final) == continuations.shape[0]

        return probs_final

    def train_epoch(
        self,
        data_loader: DataLoader,
    ):
        """
        Trains on data provided by a data loader.
        """
        for x, y in data_loader:
            self.train_batch(x, y)
