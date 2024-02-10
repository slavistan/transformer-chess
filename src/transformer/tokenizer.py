"""
The tokenizer of TAN gamelines works as follows. The following special tokens
are used in addition to the ordinary tokens encoding TAN gamelines:

    0 = padding token id
    1 = start-of-game token id
    2 = end-of-game token id

All outputs begin with the start-of-game token followed by the individually
encoded characters of the tan gameline, whose tokens are the enumerated
integers starting from 3. The second to last character, the whitespace before
the character designating the winner, is encoded as the end-of-game token.

Thus the gameline 'e4 f5 f4 g5 Qh5 W' is encoded as follows:

    [1, 19, 10, 6, 20, 11, 6, 20, 10, 6, 21, 11, 6, 26, 22, 11, 2, 3]

Note that we don't need any decoding logic, as the relevant character
sequences, i.e. the valid moves given a position, are known in advance.
"""

import numpy as np
import torch

from src.tools import lines
from src.tan_chess import (
    TANGameLine,
    TAN_GAMELINE_CHARS,
)

PADDING_TOKEN_ID = 0
START_OF_GAME_TOKEN_ID = 1
END_OF_GAME_TOKEN_ID = 2
_num_special_tokens = len((PADDING_TOKEN_ID, START_OF_GAME_TOKEN_ID, END_OF_GAME_TOKEN_ID))
_encode_gameline_dict = {c: i for i, c in enumerate(TAN_GAMELINE_CHARS, _num_special_tokens)}
WHITESPACE_TOKEN_ID = _encode_gameline_dict[" "]


def vocab_size() -> int:
    """
    Returns the vocabulary size of the tokenizer.

    The vocabulary consists of the TAN format's allowed characters representing
    a game of chess and the special tokens.
    """

    return _num_special_tokens + len(TAN_GAMELINE_CHARS)


def num_tokens(
    gameline: TANGameLine,
):
    """
    Returns the length of a tokenized gameline, not counting padding tokens.
    The length is the number of characters in the gameline plus one, accounting
    for the start-of-game token.

    Works for movelines as well.
    """

    return 1 + len(gameline)


def encode_movechars(
    tan_move_chars: str,
) -> torch.Tensor:
    """
    Tokenizes TAN move characters individually, allowing for arbitrary
    subsequences of a moveline. This will not work for gamelines.

    Returns a torch.uint8 tensor.
    """

    arr = np.empty((len(tan_move_chars),), dtype=np.uint8)
    for i, c in enumerate(tan_move_chars):
        arr[i] = _encode_gameline_dict[c]
    return torch.from_numpy(arr).type(torch.uint8)


def encode_gameline(
    tan_gameline: TANGameLine,
) -> torch.Tensor:
    """
    Encode a full TAN gameline.

    Returns a 1D torch.Tensor of size 'len(tan_gameline) + 1' and dtype
    torch.uint8.
    """

    # We're using a numpy array to fill in the encoded tokens. Random access
    # into a numpy array is much faster than into torch tensors. See
    # 'src/benchmarks/numpy_vs_torch_random_access.py'.
    width = num_tokens(tan_gameline)
    arr = np.full((width,), fill_value=PADDING_TOKEN_ID, dtype=np.uint8)
    arr[0] = START_OF_GAME_TOKEN_ID
    for i, c in enumerate(tan_gameline, 1):
        arr[i] = _encode_gameline_dict[c]
    arr[-2] = END_OF_GAME_TOKEN_ID

    # The tensor created by torch.from_numpy() inherits the appropriate dtype
    # from the numpy array. This will produce a torch.uint8 tensor.
    return torch.from_numpy(arr)


def encode_tan_file(
    path: str,
    context_size: int,
) -> torch.Tensor:
    """
    Tokenizes all games in a .tan file whose token sequences fit in a tensor of
    width 'context_size + 1'.

    Utility function used to create training data tensors.
    """

    # Tensor width and height. 'lines()' counts the newline character of a line
    # when comparing its length against 'max_len', which adds the offset we
    # need to account for the start-of-game token.
    width = context_size + 1
    height = sum((1 for _ in lines(path, max_len=width)))

    # Initialize tensor with the padding token's id.
    data = torch.full((height, width), fill_value=PADDING_TOKEN_ID, dtype=torch.uint8)
    for i, line in enumerate(lines(path, max_len=width)):
        gameline = line.rstrip()
        encd = encode_gameline(gameline)
        data[i, : len(encd)] = encd

        if (i + 1) % 10000 == 0:
            msg = f"Loading games to memory: {i+1}/{height}"
            print(msg, end="\b" * len(msg), flush=True)

    return data
