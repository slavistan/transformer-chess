import mmap
import os
import sys
import torch
import contextlib


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


@contextlib.contextmanager
def unaligned_ro_mmap_open(file_path: str, length: int, offset: int = 0):
    file_sz = os.path.getsize(file_path)
    if offset + length > file_sz:
        raise ValueError("Requested portion exceeds file size")
    with open(file_path, "rb") as f:
        mm = RoUnalignedMMAP(f.fileno(), length, offset)
    try:
        yield mm
    finally:
        mm.close()


class RoUnalignedMMAP:
    """
    A read-only memory-mapped file that allows unaligned access. Wraps a regular mmap object.
    """
    def __init__(self, fileno, length, offset=0):
        aligned_offset = (offset // mmap.PAGESIZE) * mmap.PAGESIZE
        self.manual_offset = offset - aligned_offset
        self.mm = mmap.mmap(
            fileno,
            length=length + self.manual_offset,
            access=mmap.ACCESS_READ,
            offset=aligned_offset,
        )
        self.mm.seek(self.manual_offset)

    def seek(self, offset, whence=0):
        if whence == 0:
            self.mm.seek(offset + self.manual_offset)
        else:
            self.mm.seek(offset, whence)

    def tell(self):
        return self.mm.tell() - self.manual_offset

    def find(self, sub, start=0, stop=sys.maxsize):
        mm_find_index = self.mm.find(sub, start, stop)
        return mm_find_index - self.manual_offset if mm_find_index != -1 else -1

    def rfind(self, sub, start=0, stop=sys.maxsize):
        mm_find_index = self.mm.rfind(sub, start, stop)
        return mm_find_index - self.manual_offset if mm_find_index != -1 else -1

    def readline(self):
        return self.mm.readline()

    def read(self, n: int | None = None):
        return self.mm.read(n)

    def size(self):
        return self.mm.size()

    def close(self):
        return self.mm.close()

    @property
    def closed(self):
        return self.mm.closed
