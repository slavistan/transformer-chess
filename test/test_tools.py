from __future__ import annotations

from itertools import accumulate
import os
import itertools
from pathlib import Path
import array
import multiprocessing as mp

import pytest

from src.tools import *
from src.db_utils import (
    splitfn_chunk,
    processfn_find_lines,
    collectfn_get_lines,
    splitfn_lines,
    pgn_to_tan,
    pgn_gameline_to_tan,
)

utf8_file = "test/assets/utf-8.txt"
assert os.path.getsize(utf8_file) == 581  # single page
empty_file = "./test/assets/empty.txt"
assert os.path.getsize(empty_file) == 0
pgn_file = "data/example.pgn"
assert os.path.getsize(pgn_file) == 1140613  # multi page
files = [utf8_file, pgn_file]
files0 = files + [empty_file]


class Test_unaligned_ro_mmap_open:
    assert unaligned_ro_mmap_open.__name__ == "unaligned_ro_mmap_open"

    def test_map_empty_file(self):
        with pytest.raises(ValueError):
            file_sz = os.path.getsize(empty_file)
            with unaligned_ro_mmap_open(empty_file, file_sz):
                pass

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


class Test_splitfn_chunk:
    assert splitfn_chunk.__name__ == "splitfn_chunk"

    @pytest.mark.parametrize("file_path,num_workers", list(itertools.product(files + [empty_file], list(range(1, mp.cpu_count())) + [1024, 4096, 16384] + [os.path.getsize(f) for f in files])))
    def test_happy_path(self, file_path, num_workers):
        file_sz = os.path.getsize(file_path)
        chunks = splitfn_chunk(file_path, num_workers)

        offsets, lengths = chunks[::2], chunks[1::2]
        assert len(offsets) == num_workers
        assert sorted(offsets) == list(offsets)
        assert len(lengths) == num_workers
        assert sum(lengths) == file_sz


class Test_processfn_find_lines:
    assert processfn_find_lines.__name__ == "processfn_find_lines"

    @pytest.mark.parametrize("file_path", files + [empty_file])
    def test_happy_path(self, file_path):
        with open(file_path, "rb") as f, open(file_path, "rb") as f2:
            file_sz = os.path.getsize(file_path)
            lines_info, _ = processfn_find_lines(f)
            lines = f2.readlines()

            offsets, lengths = list(lines_info[::2]), list(lines_info[1::2])
            assert len(offsets) == len(lengths)
            assert len(offsets) == len(lines)
            assert all([l > 0 for l in lengths])
            assert sum(lengths) == file_sz
            assert sorted(offsets) == offsets

            offset = 0
            for i, line in enumerate(lines):
                assert offsets[i] == offset
                assert lengths[i] == len(line)
                offset += len(line)

    def test_newline(self):
        with unaligned_ro_mmap_open(utf8_file, 52, 2 * 52) as mm:
            lines_info_arr, ends_in_newline = processfn_find_lines(mm)
            assert ends_in_newline
            have_lengths = list(lines_info_arr[1::2])
            want_lengths = [51, 1]
            assert have_lengths == want_lengths


class Test_collectfn_get_lines:
    assert collectfn_get_lines.__name__ == "collectfn_get_lines"

    @staticmethod
    def expand(lines_info):
        """
        Expands tests data representing lines only as their lengths, to match
        the input of a CollectFn. Distinguishes between line lengths with the
        end-of-line flag or simple line lengths used to expand the expected
        outputs.
        """

        result = []
        if isinstance(lines_info[0], tuple):
            for lengths, ends_in_newline in lines_info:
                offsets = accumulate(lengths, initial=0)
                # interweave offsets and lengths to arrive at out required
                # format.
                encoded = [val for pair in zip(offsets, lengths) for val in pair]
                result.append([(array.array("Q", encoded), ends_in_newline)])
        else:
            offsets = accumulate(lines_info, initial=0)
            result = [val for pair in zip(offsets, lines_info) for val in pair]
        return result

    # Legend:
    #   | chunk boundaries
    #   n newline
    #   ! newline on chunk boundary
    def test_default(self):
        # |----|
        lengths = [
            ([10], False),
        ]
        chunk_results = Test_collectfn_get_lines.expand(lengths)
        have = list(collectfn_get_lines(chunk_results))
        want = [10]
        assert have == Test_collectfn_get_lines.expand(want)

        # |----!
        lengths = [
            ([10], False),
        ]
        chunk_results = Test_collectfn_get_lines.expand(lengths)
        have = list(collectfn_get_lines(chunk_results))
        want = [10]
        assert have == Test_collectfn_get_lines.expand(want)

        # |--n-|
        lengths = [
            ([10, 5], False),
        ]
        chunk_results = Test_collectfn_get_lines.expand(lengths)
        have = list(collectfn_get_lines(chunk_results))
        want = [10, 5]
        assert have == Test_collectfn_get_lines.expand(want)

        # |--n-!
        lengths = [
            ([10, 5], True),
        ]
        chunk_results = Test_collectfn_get_lines.expand(lengths)
        have = list(collectfn_get_lines(chunk_results))
        want = [10, 5]
        assert have == Test_collectfn_get_lines.expand(want)

        # |-n-n-!
        lengths = [
            ([5, 5, 5], True),
        ]
        chunk_results = Test_collectfn_get_lines.expand(lengths)
        have = list(collectfn_get_lines(chunk_results))
        want = [5, 5, 5]
        assert have == Test_collectfn_get_lines.expand(want)

        # |-n-|-n-|
        lengths = [
            ([5, 5], False),
            ([3, 3], False),
        ]
        chunk_results = Test_collectfn_get_lines.expand(lengths)
        have = list(collectfn_get_lines(chunk_results))
        want = [5, 8, 3]
        assert have == Test_collectfn_get_lines.expand(want)

        # |-n-!-n-|
        lengths = [
            ([5, 5], True),
            ([3, 3], False),
        ]
        chunk_results = Test_collectfn_get_lines.expand(lengths)
        have = list(collectfn_get_lines(chunk_results))
        want = [5, 5, 3, 3]
        assert have == Test_collectfn_get_lines.expand(want)

        # |-n-!-n-!
        lengths = [
            ([5, 5], True),
            ([3, 3], True),
        ]
        chunk_results = Test_collectfn_get_lines.expand(lengths)
        have = list(collectfn_get_lines(chunk_results))
        want = [5, 5, 3, 3]
        assert have == Test_collectfn_get_lines.expand(want)

        # |---!-n-!
        lengths = [
            ([5], True),
            ([3, 3], True),
        ]
        chunk_results = Test_collectfn_get_lines.expand(lengths)
        have = list(collectfn_get_lines(chunk_results))
        want = [5, 3, 3]
        assert have == Test_collectfn_get_lines.expand(want)

        # |-n-|---|
        lengths = [
            ([3, 4], False),
            ([5], False),
        ]
        chunk_results = Test_collectfn_get_lines.expand(lengths)
        have = list(collectfn_get_lines(chunk_results))
        want = [3, 9]
        assert have == Test_collectfn_get_lines.expand(want)

        # |-n-!---|
        lengths = [
            ([3, 4], True),
            ([5], False),
        ]
        chunk_results = Test_collectfn_get_lines.expand(lengths)
        have = list(collectfn_get_lines(chunk_results))
        want = [3, 4, 5]
        assert have == Test_collectfn_get_lines.expand(want)

        # |-n-!---!
        lengths = [
            ([3, 4], True),
            ([5], True),
        ]
        chunk_results = Test_collectfn_get_lines.expand(lengths)
        have = list(collectfn_get_lines(chunk_results))
        want = [3, 4, 5]
        assert have == Test_collectfn_get_lines.expand(want)

        # |-n-|---|--n-|
        lengths = [
            ([3, 4], False),
            ([5], False),
            ([6, 7], False),
        ]
        chunk_results = Test_collectfn_get_lines.expand(lengths)
        have = list(collectfn_get_lines(chunk_results))
        want = [3, 15, 7]
        assert have == Test_collectfn_get_lines.expand(want)

        # |-n-|---!--n-|
        lengths = [
            ([3, 4], False),
            ([5], True),
            ([6, 7], False),
        ]
        chunk_results = Test_collectfn_get_lines.expand(lengths)
        have = list(collectfn_get_lines(chunk_results))
        want = [3, 9, 6, 7]
        assert have == Test_collectfn_get_lines.expand(want)

        # |-n-n-|---|-n-n-|
        lengths = [
            ([1, 2, 3], False),
            ([4], False),
            ([5, 6, 7], False),
        ]
        chunk_results = Test_collectfn_get_lines.expand(lengths)
        have = list(collectfn_get_lines(chunk_results))
        want = [1, 2, 12, 6, 7]
        assert have == Test_collectfn_get_lines.expand(want)

        # |-n-n-|---|---|-n-n-|
        lengths = [
            ([1, 2, 3], False),
            ([4], False),
            ([5], False),
            ([6, 7, 8], False),
        ]
        chunk_results = Test_collectfn_get_lines.expand(lengths)
        have = list(collectfn_get_lines(chunk_results))
        want = [1, 2, 3 + 4 + 5 + 6, 7, 8]
        assert have == Test_collectfn_get_lines.expand(want)

        # |-n-n-|---|---!-n-n-|
        lengths = [
            ([1, 2, 3], False),
            ([4], False),
            ([5], True),
            ([6, 7, 8], False),
        ]
        chunk_results = Test_collectfn_get_lines.expand(lengths)
        have = list(collectfn_get_lines(chunk_results))
        want = [1, 2, 3 + 4 + 5, 6, 7, 8]
        assert have == Test_collectfn_get_lines.expand(want)

        # |-n-n-|---!---|-n-n-|
        lengths = [
            ([1, 2, 3], False),
            ([4], True),
            ([5], False),
            ([6, 7, 8], False),
        ]
        chunk_results = Test_collectfn_get_lines.expand(lengths)
        have = list(collectfn_get_lines(chunk_results))
        want = [1, 2, 3 + 4, 5 + 6, 7, 8]
        assert have == Test_collectfn_get_lines.expand(want)

        # |--!!|
        lengths = [
            ([2], True),
            ([1], True),
            ([1], False),
        ]
        chunk_results = Test_collectfn_get_lines.expand(lengths)
        have = list(collectfn_get_lines(chunk_results))
        want = [2, 1, 1]
        assert have == Test_collectfn_get_lines.expand(want)

        # |--||
        lengths = [
            ([2], False),
            ([1], False),
        ]
        chunk_results = Test_collectfn_get_lines.expand(lengths)
        have = list(collectfn_get_lines(chunk_results))
        want = [3]
        assert have == Test_collectfn_get_lines.expand(want)

        # |--|||--|-|
        lengths = [
            ([2], False),
            ([1], False),
            ([1], False),
            ([2], False),
            ([2], False),
        ]
        chunk_results = Test_collectfn_get_lines.expand(lengths)
        have = list(collectfn_get_lines(chunk_results))
        want = [8]
        assert have == Test_collectfn_get_lines.expand(want)


class Test_splitfn_lines:
    assert splitfn_lines.__name__ == "splitfn_lines"

    num_workers = set([1, 2, 4, mp.cpu_count(), mp.cpu_count() + 1])

    @pytest.mark.parametrize("num_workers", num_workers)
    def test_empty_file(self, num_workers):
        lines_info_arr = splitfn_lines(empty_file, num_workers=num_workers, quiet=True)
        have_lengths = list(lines_info_arr[1::2])
        want_lengths = []
        assert want_lengths == have_lengths

    @pytest.mark.parametrize("file_path,num_workers", list(itertools.product(files, num_workers)))
    def test_default(self, file_path, num_workers):
        with open(file_path, "rb") as f:
            lines = f.readlines()

        lines_info_arr = splitfn_lines(file_path, num_workers=num_workers, quiet=True)
        have = list(lines_info_arr[1::2])
        want = [len(l) for l in lines]
        assert len(have) == len(have)
        assert have == want


class Test_pgn_to_tan:
    assert pgn_to_tan.__name__ == "pgn_to_tan"

    @pytest.mark.parametrize("num_workers", [1, 2, 11, mp.cpu_count()])
    def test_default(self, num_workers, tmp_path: Path):
        with open(pgn_file, "r") as f:
            pgn_gamelines = [l for l in f.readlines() if l.startswith("1")]
            tan_gamelines = [pgn_gameline_to_tan(l) + "\n" for l in pgn_gamelines]

        out_file = str(tmp_path / "out.pgn")
        pgn_to_tan(pgn_file, out_file=out_file, parallel_process_args={"num_workers": num_workers, "quiet": True})
        with open(out_file, "r") as f:
            have = f.readlines()
        want = tan_gamelines

        assert len(want) == len(have)
        assert want == have
