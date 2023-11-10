import sys
import torch

def count_lines_in_file(file_path: str, *, max_lines=sys.maxsize) -> int:
    """Counts the number of lines in a file. Returns early, if max_lines is reached."""

    i = 0
    with open(file_path, "r") as f:
        for _ in f:
            if i >= max_lines:
                break
            i += 1
    return i


def torch_elem_size(dtype: torch.dtype) -> int:
    """
    Returns the size in bytes of a single element of the given torch dtype.
    """

    t = torch.empty((1,), dtype=dtype)
    return t.element_size()