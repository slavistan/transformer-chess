from __future__ import annotations

import torch
from torch.utils.data import Dataset


class RAMDump(Dataset):
    """
    Pytorch dataset which loads the training data into main memory as a single
    tensor.
    """

    # torch.uint8 data tensor containing tokenized gamelines. We use uint8 here
    # to save memory in order to fit a large database of games into main
    # memory.
    data: torch.Tensor

    def __init__(
        self,
        data: torch.Tensor | str,
    ):
        """
        Loads data from either an existing tensor or from a file created
        previously by 'torch.save()'.
        """

        if isinstance(data, str):
            self.data = torch.load(data)
        else:
            self.data = data
        if self.data.dtype != torch.uint8:
            self.data = self.data.type(torch.uint8)

    def __getitem__(self, idx):
        # Convert to proper torch integer type, as torch.uint8 is treated as
        # torch.bool, which will attempt to use boolean indexing, instead of
        # treating the values as indices into the embedding matrix.
        t = self.data.to(torch.int)
        return t[idx, :-1], t[idx, 1:]

    def __len__(self):
        return len(self.data)
