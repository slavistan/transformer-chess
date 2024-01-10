import importlib
import pkgutil
from typing import Union, IO
import mmap
import os
import sys
import contextlib

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


@contextlib.contextmanager
def unaligned_ro_mmap_open(input_file: Union[str, IO], length: int, offset: int = 0):
    """
    Context manager for a RoUnalignedMMAP. Accepts a file path or opened file
    handle.
    """

    file_sz = os.path.getsize(input_file) if isinstance(input_file, str) else os.fstat(input_file.fileno()).st_size
    if offset + length > file_sz:
        raise ValueError("Requested portion exceeds file size")

    if isinstance(input_file, str):
        with open(input_file, "rb") as f:
            mm = RoUnalignedMMAP(f.fileno(), length, offset)
    else:
        mm = RoUnalignedMMAP(input_file.fileno(), length, offset)
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


def import_all_modules(package_name: str):
    """
    Recusively imports all modules from a root package.
    """

    package = importlib.import_module(package_name)
    for _, module_name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_module_name = f"{package_name}.{module_name}"
        importlib.import_module(full_module_name)
        if is_pkg:
            import_all_modules(full_module_name)
