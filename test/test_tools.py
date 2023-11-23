from __future__ import annotations
import os
import itertools
import pytest
from typing import Callable, Type, Any
import pytest
from src.tools import *
from src.db_utils import splitfn_chunk

utf8_file = "test/assets/utf-8.txt"  # small file (581 bytes) fitting into a single page
pgn_file = "data/example.pgn"  # large file (1140613 bytes) spanning multiple pages
files = [utf8_file, pgn_file]


class TestUnalignedRoMmapOpen:
    @pytest.mark.parametrize("file_path", files)
    def test_map_full_file_read_all(self, file_path: str):
        """Mapping the full file should yield the same result as reading the file."""
        file_sz = os.path.getsize(file_path)
        with unaligned_ro_mmap_open(file_path, file_sz) as mm, open(file_path, "rb") as f:
            assert mm.size() == file_sz
            have = mm.read()
            want = f.read()
            assert len(have) == len(want)
            assert have == want

    @pytest.mark.parametrize("file_path", files)
    def test_map_start_of_file_read_all(self, file_path):
        file_sz = os.path.getsize(file_path)
        num_bytes = 16
        with unaligned_ro_mmap_open(file_path, num_bytes) as mm, open(file_path, "rb") as f:
            have = mm.read()
            want = f.read(num_bytes)
            assert len(have) == len(want)
            assert have == want
            assert file_sz == mm.size()

    @pytest.mark.parametrize("file_path", files)
    def test_map_end_of_file_end_read_all(self, file_path):
        num_bytes = 16
        file_sz = os.path.getsize(file_path)
        with unaligned_ro_mmap_open(file_path, num_bytes, file_sz - num_bytes) as mm, open(file_path, "rb") as f:
            have = mm.read()
            want = f.read()[-num_bytes:]
            assert len(have) == len(want)
            assert have == want

    @pytest.mark.parametrize("file_path,n", list(itertools.product(files, range(0, 32, 8))))
    def test_map_middle_of_file_read_all(self, file_path, n):
        file_sz = os.path.getsize(file_path)
        offset = size = 2**n
        if offset + size > file_sz:
            return
        with unaligned_ro_mmap_open(file_path, size, offset) as mm, open(file_path, "rb") as f:
            have = mm.read()
            want = f.read()[offset : offset + size]
            assert len(have) == len(want)
            assert have == want

    @pytest.mark.parametrize("file_path", files)
    def test_readline_over_full_file(self, file_path):
        file_sz = os.path.getsize(file_path)
        with unaligned_ro_mmap_open(file_path, file_sz) as mm, open(file_path, "rb") as f:
            line = mm.readline()
            while line:
                have = line
                want = f.readline()
                assert have == want
                line = mm.readline()
            assert mm.readline() == b""
            assert mm.tell() == file_sz
            with pytest.raises(ValueError):
                mm.seek(1, 1)

    @pytest.mark.parametrize("file_path", files)
    def test_readline_partial_line(self, file_path):
        file_sz = os.path.getsize(file_path)
        line_offset = 2
        with unaligned_ro_mmap_open(file_path, file_sz - line_offset, line_offset) as mm, open(file_path, "rb") as f:
            partial_line = mm.readline()
            full_line = f.readline()
            assert len(partial_line) == len(full_line) - line_offset
            assert partial_line == full_line[line_offset:]
            assert mm.readline() == f.readline()

    @pytest.mark.parametrize("file_path", files)
    def test_size(self, file_path):
        file_sz = os.path.getsize(file_path)
        with unaligned_ro_mmap_open(file_path, file_sz) as mm:
            assert mm.size() == file_sz
        with unaligned_ro_mmap_open(file_path, 16, 77) as mm:
            assert mm.size() == file_sz
        with unaligned_ro_mmap_open(file_path, file_sz - 16, 16) as mm:
            assert mm.size() == file_sz

    @pytest.mark.parametrize("file_path,n", list(itertools.product(files, [1, 32, 1024])))
    def test_excessive_range(self, file_path, n):
        file_sz = os.path.getsize(file_path)
        with pytest.raises(ValueError):
            with unaligned_ro_mmap_open(file_path, file_sz, n):
                pass

    def test_find(self):
        file_path = utf8_file
        file_sz = os.path.getsize(file_path)
        mask_a = b"Chess"
        mask_b = b"Chess II"
        with open(file_path, "rb") as f:
            data = f.read()
            mask_a_offset = data.find(mask_a)
            mask_b_offset = data.find(mask_b)

        with unaligned_ro_mmap_open(file_path, file_sz - mask_a_offset, mask_a_offset) as mm:
            assert mm.tell() == 0
            have = mm.find(mask_a)
            want = 0
            assert have == want
            assert mm.tell() == 0

            have = mm.find(mask_b)
            want = mask_b_offset - mask_a_offset
            assert have == want
            assert mm.tell() == 0

        with unaligned_ro_mmap_open(file_path, file_sz) as mm:
            have = mm.find(mask_a, mask_a_offset)
            want = mask_a_offset
            assert have == want

            have = mm.find(mask_a, mask_a_offset + 1)
            want = mask_b_offset
            assert have == want

    def test_rfind(self):
        file_path = utf8_file
        file_sz = os.path.getsize(file_path)
        mask_a = b"Chess"
        mask_b = b"Chess II"
        with open(file_path, "rb") as f:
            data = f.read()
            mask_a_offset = data.find(mask_a)
            mask_b_offset = data.find(mask_b)

        with unaligned_ro_mmap_open(file_path, file_sz - mask_a_offset, mask_a_offset) as mm:
            assert mm.tell() == 0
            have = mm.rfind(mask_a)
            want = mask_b_offset - mask_a_offset
            assert have == want
            assert mm.tell() == 0

            have = mm.rfind(mask_b)
            want = mask_b_offset - mask_a_offset
            assert have == want
            assert mm.tell() == 0

            have = mm.rfind(mask_a, 0, 128)
            want = 0
            assert have == want
            assert mm.tell() == 0

        with unaligned_ro_mmap_open(file_path, file_sz) as mm:
            have = mm.rfind(mask_a, 0, 128)
            want = mask_a_offset
            assert have == want
            assert mm.tell() == 0

    def test_seek(self):
        file_path = utf8_file
        file_sz = os.path.getsize(file_path)
        mask_a = b"Chess"
        mask_b = b"Chess II"
        with open(file_path, "rb") as f:
            data = f.read()
            line0 = data.splitlines()[0] + b"\n"
            mask_a_offset = data.find(mask_a)
            mask_b_offset = data.find(mask_b)

        with unaligned_ro_mmap_open(file_path, file_sz) as mm:
            mm.seek(mask_a_offset)
            have = mm.read(len(mask_a))
            want = mask_a
            assert have == want

            mm.seek(mask_b_offset - mask_a_offset - len(mask_a), 1)
            have = mm.read(len(mask_b))
            want = mask_b
            assert have == want

            mm.seek(0)
            want = line0
            have = mm.readline()
            assert have == want

    def test_closed(self):
        with open(utf8_file, "rb") as f:
            mm = RoUnalignedMMAP(f.fileno(), 128, 16)
            assert not mm.closed
            mm.read(14)
            assert not mm.closed
            mm.close()
            assert mm.closed

    def test_mmap_from_opened_file(self):
        file_path = utf8_file
        file_sz = os.path.getsize(file_path)
        mask_a = b"Chess"
        mask_b = b"Chess II"
        with open(file_path, "rb") as f:
            data = f.read()
            mask_a_offset = data.find(mask_a)
            mask_b_offset = data.find(mask_b)

            with unaligned_ro_mmap_open(f, file_sz - mask_a_offset, mask_a_offset) as mm:
                assert mm.tell() == 0
                have = mm.rfind(mask_a)
                want = mask_b_offset - mask_a_offset
                assert have == want
                assert mm.tell() == 0

                have = mm.rfind(mask_b)
                want = mask_b_offset - mask_a_offset
                assert have == want
                assert mm.tell() == 0

                have = mm.rfind(mask_a, 0, 128)
                want = 0
                assert have == want
                assert mm.tell() == 0

            with unaligned_ro_mmap_open(f, file_sz) as mm:
                have = mm.rfind(mask_a, 0, 128)
                want = mask_a_offset
                assert have == want
                assert mm.tell() == 0


class TestTypeofArg0:
    def test_no_arguments(self):
        def func():
            pass

        assert typeof_arg0(func) is None

    def test_no_type_hint(self):
        def func(arg):
            return arg

        assert typeof_arg0(func) is Any

    def test_with_type_hint(self):
        def func(arg: int):
            return arg

        assert typeof_arg0(func) is int

    def test_multiple_arguments(self):
        def func(arg1, arg2: str):
            return (arg1, arg2)

        assert typeof_arg0(func) is Any


class Test_splitfn_chunk:
    assert splitfn_chunk.__name__ == "splitfn_chunk"

    @pytest.mark.parametrize("file_path,num_workers", list(itertools.product(files, [1, 32, 1024, 4096, 16384] + [os.path.getsize(f) for f in files])))
    def test_default(self, file_path, num_workers):
        file_sz = os.path.getsize(file_path)
        chunks = splitfn_chunk(file_path, num_workers)

        offsets, lengths = chunks[::2], chunks[1::2]
        assert len(offsets) == num_workers
        assert sorted(offsets) == list(offsets)
        assert len(lengths) == num_workers
        assert sum(lengths) == file_sz

