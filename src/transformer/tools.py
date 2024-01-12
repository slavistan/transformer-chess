from typing import Sequence

import numpy as np
import torch

from src.tan_chess import TAN_MOVELINE_CHARS

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
