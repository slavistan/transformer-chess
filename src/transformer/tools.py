from typing import Sequence

import numpy as np
import torch

from src.tan_chess import (
    TANMove,
    TANGameLine,
    TAN_GAMELINE_CHARS,
)

PADDING_TOKEN_ID = 0
START_OF_GAME_TOKEN_ID = 1
END_OF_GAME_TOKEN_ID = 2
num_special_tokens = len((PADDING_TOKEN_ID, START_OF_GAME_TOKEN_ID, END_OF_GAME_TOKEN_ID))
encode_gameline_dict = {c: i for i, c in enumerate(TAN_GAMELINE_CHARS, num_special_tokens)}
decode_gameline_dict = {i: c for c, i in encode_gameline_dict.items()}
WHITESPACE_TOKEN_ID = encode_gameline_dict[" "]


def encode_gameline_as_np8(
    tan_gameline: TANGameLine,
) -> np.ndarray:
    """
    Returns a 1d numpy array of size `len(tan_gameline) + 1`, which includes the
    start-of-game token.
    """

    width = 1 + len(tan_gameline)
    result = np.empty((width,), dtype=np.uint8)
    result[0] = START_OF_GAME_TOKEN_ID
    for i, c in enumerate(tan_gameline, 1):
        result[i] = encode_gameline_dict[c]
    result[-2] = END_OF_GAME_TOKEN_ID
    return result


def encode_move(
    tan_move: TANMove,
) -> torch.Tensor:
    """
    Encodes a single move.

    The move may contain whitespaces, which will be encoded correctly.
    """

    arr = np.empty((len(tan_move),), dtype=np.uint8)
    for i, c in enumerate(tan_move):
        arr[i] = encode_gameline_dict[c]
    return torch.from_numpy(arr)


def encode_gameline(
    tan_gameline: TANGameLine,
    *,
    pad_to_length=0,
) -> torch.Tensor:
    """
    Encode a full TAN gameline.

    Returns a torch tensor of shape `(max(len(tan_gameline + 1, pad_to_length)), )` and
    dtype torch.uint8.
    """

    # We're using a numpy array to fill in the encoded tokens. Random access
    # into a numpy array is much faster than into torch tensors. See
    # 'src/benchmarks/numpy_vs_toch_random_access.py'.
    width = max(1 + len(tan_gameline), pad_to_length)
    arr = np.full((width,), fill_value=PADDING_TOKEN_ID, dtype=np.uint8)
    arr[0] = START_OF_GAME_TOKEN_ID
    for i, c in enumerate(tan_gameline, 1):
        arr[i] = encode_gameline_dict[c]
    arr[-2] = END_OF_GAME_TOKEN_ID

    # The tensor created by torch.from_numpy() inherits the appropriate dtype
    # from the numpy array. This will produce a torch.uint8 tensor.
    return torch.from_numpy(arr)


def decode_move(tan_tokens: torch.Tensor) -> str:
    """
    Decodes a single encoded move in TAN format.
    """

    return "".join([decode_gameline_dict[np.uint8(t)] for t in tan_tokens.cpu()])


def decode_moveline(tan_tokens: Sequence[int]) -> str:
    return "".join([decode_gameline_dict[np.uint8(t)] for t in tan_tokens])
