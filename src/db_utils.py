"""Functionality to transform lichess chess games database files avaliable at
https://database.lichess.org/"""

import array
import time
import zstandard
import io
import os
import re
import math
import torch
import chess
import multiprocessing as mp
import logging
import psutil
from typing import List, Dict, Tuple, Callable, Any, Collection, BinaryIO, Sequence, Type
from tqdm.auto import tqdm
from src.san_chess import Outcome, get_outcome, SAN_ANNOTATION_POSTFIX
from src.tools import RoUnalignedMMAP, unaligned_ro_mmap_open

# String to array of ints.
# def encode_tan_movechars(tan_movechars):
#     return tuple(atoi_tan_movechars[c] for c in tan_movechars)
#     return movetext


# def tan_to_tensor(
#     tan_file_path,
#     width,
#     *,
#     max_games=1e9,
#     boundary_token="%",
#     dtype=torch.long,
#     device="cpu",
# ):
#     re_rm_eog = re.compile(" (" + "|".join(TAN_MOVELINE_CHARS) + ")$")
#     filter_fn = (
#         lambda tan_movetext: len(re.sub(re_rm_eog, "", tan_movetext)) - 2 <= width
#     )


def tan_gamelines_from_pgn_zstd(db_file: str, out_file: str | None = None):
    with open(db_file, "rb") as f_in:
        dctx = zstandard.ZstdDecompressor(max_window_size=2147483648)
        stream_reader = dctx.stream_reader(f_in)
        text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
        for line in text_stream:
            # Movetexts begin '1. ', the move indicator for the first move.
            if line.startswith("1. "):
                # Skip incomplete games denoted by an asterisk at the end of
                # the movetext.
                line = line.rstrip()
                if line[-1] == "*":
                    continue
                line = pgn_gameline_to_tan(line)
                yield line


# def extract_pgn_movetexts(database, to_tan=True, skip_incomplete=True):
#     """Extracts the movetexts in trimmed algebraic notation from an zst-encoded
#     database of chess games in PGN format.

#     This function returns a generator which yields one full game's movetext per
#     output. The outputs do not contain a trailing newline."""
#     with open(database, "rb") as fh:
#         dctx = zstandard.ZstdDecompressor(max_window_size=2147483648)
#         stream_reader = dctx.stream_reader(fh)
#         text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
#         text_stream = io.TextIOWrapper(stream_reader, encoding= utf-8 )for line in text_stream:
#             line = line.rstrip()
#             # Movetexts begin '1. ', the move indicator for the first move.
#             if line.startswith("1. "):
#                 # Skip incomplete games denoted by an asterisk at the end of
#                 # the movetext.
#                 if skip_incomplete and line.rfind("*") != -1:
#                     continue
#                 if to_tan:
#                     line = san_gameline_to_tan(line)
#                 yield line

#                 data[i, 1 : len(tokens) + 1] = tokens


_san_movetext_re = [
    # Matches move indices. In addition to regular move indices, some moves
    # are annotated with engine evaluations and use use a triple period to
    # designate the continuation of the movetext:
    #
    #   1. b4 { [%eval -0.46] } 1... d5 { [%eval -0.44] } 2. Bb2 ...
    re.compile(r"[1-9][0-9]*\.+\s?"),
    # Matches annotations in brackets {}.
    re.compile(r"\{[^}]*\}\s?"),
    # Matches move quality annotations such as blunders or brilliant moves, and
    # checks and checkmate indicators.
    re.compile(f"[{SAN_ANNOTATION_POSTFIX}]"),
]

_san_movetext_eog_to_tan = {"1-0": "W", "0-1": "S", "1/2-1/2": "U", " *": ""}  # incomplete games are denoted by an asterisk


def pgn_gameline_to_tan(san_gameline: str) -> str:
    """Converts a game's PGN's movetext (aka. gameline) to trimmed algebraic
    notation. Expects a string representing a full game's movetext in algebraic
    notation. Trailing newlines will be stripped."""

    san_gameline = san_gameline.rstrip()
    for rgx in _san_movetext_re:
        san_gameline = re.sub(rgx, "", san_gameline)

    for san_eog, tan_eog in _san_movetext_eog_to_tan.items():
        if san_gameline.endswith(san_eog):
            san_gameline = san_gameline[: -len(san_eog)] + tan_eog
            break

    return san_gameline


def pgn_to_tan(pgn_file: str, out_file: str, parallel_process_args={}):
    return parallel_process(
        pgn_file,
        split_fn=splitfn_pgn_gameslines,
        process_fn=processfn_pgn_gameline_to_tan,
        collect_fn=make_writefn(out_file),
        **parallel_process_args,
    )


# TODO: Parallel process nach tools verschieben
SplitFn = Callable[[str], Sequence[int]]
ProcessFn = Callable[[bytes | RoUnalignedMMAP, Tuple], Any | None]
CollectFn = Callable[[List[List[Any]]], Any]


def parallel_process(
    in_file: str,
    # TODO: Offset-Größe Tuples erlauben (Callable[[str], Sequence[int] | Sequence[Tuple[int, int]]],
    split_fn: SplitFn,
    process_fn: ProcessFn,
    collect_fn: CollectFn,
    *,
    process_fn_extra_args: Tuple = (),
    num_workers: int = 0,
    quiet=False,
    mmap_mode=False,
    logger=logging.getLogger(),
):
    if num_workers == 0:
        num_workers = mp.cpu_count()

    # Identify chunks to be processed. A chunk is identified by its byte offset
    # and its size in bytes. For better data locality chunks should be sorted
    # by their offsets. Chunks don't have to be contiguous or non-overlapping.
    #
    # Each contiguous pair of elements encodes one chunk. E.g.
    #   [0, 14, 14, 7, 23, 4, 19, 4]
    #    ^  ^^
    #    ^  ^^~~~ size in bytes of chunk 0
    #    ^~~~~~~~ offset in bytes of chunk 0
    logger.info(f"{parallel_process.__name__}: Calling splitfn ...")
    chunks_info = split_fn(in_file)
    logger.info(f"{parallel_process.__name__}: done.")
    num_chunks = len(chunks_info) // 2

    logger.info(f"{parallel_process.__name__}: Preparing workers' data ...")
    chunks_per_worker = math.ceil(num_chunks / num_workers)
    worker_args: List[Tuple[str, Sequence[int], ProcessFn, int, Tuple, bool, bool]] = []
    for i in range(num_workers):
        worker_chunks_info = chunks_info[i * chunks_per_worker * 2 : i * chunks_per_worker * 2 + chunks_per_worker * 2]
        worker_args.append(
            (
                in_file,  # file path
                worker_chunks_info,  # chunks to be processed by the worker
                process_fn,  # chunk processing function
                i,  # worker index
                # TODO: Kann lokale Funktion mit Closure mglw. in temporäre Datei geschrieben werden?
                #       nach Collect() kann diese wieder gelöscht werden. Mal testen.
                process_fn_extra_args,  # additional arguments for processfn
                quiet,  # verbosity flag
                mmap_mode,
            )
        )
    logger.info(f"{parallel_process.__name__}: done.")

    # Start parallel processing and collect the results.
    logger.info(f"{parallel_process.__name__}: Starting parallel processing with {num_workers} workers ...")
    with mp.get_context("spawn").Pool(num_workers) as p:
        worker_results: List[List[Any]] = p.starmap(worker, worker_args)
    logger.info(f"{parallel_process.__name__}: done.")

    # Process the results.
    logger.info(f"{parallel_process.__name__}: Calling collectfn ...")
    result = collect_fn(worker_results)
    logger.info(f"{parallel_process.__name__}: done.")
    return result


def worker(
    in_file: str,
    chunks_info: Sequence[int],
    process_fn: ProcessFn,
    worker_idx: int,
    process_fn_extra_args: Tuple[Any],
    quiet: bool,
    mmap_mode: bool,
) -> List[Any]:
    """
    Sequentially processes chunks of data from a file. This function is spawned
    by `parallel_process` and should not be called directly.

    Args:
        in_file: Path to the input file.
        chunks_info: List of chunk offsets and sizes.
        process_fn: Function to process each chunk of data.
        worker_idx: Index of the current worker.
        process_fn_extra_args: Extra arguments to pass to the process function.
        quiet: Whether to suppress progress bar output.

    Returns:
        List[Any]: List of results from processing each chunk of data.
    """

    # Set up progress bar.
    if not quiet:
        work = sum(chunks_info[1::2])
        pb = tqdm(
            total=work,
            unit="B",
            unit_scale=True,
            desc=f"Worker {worker_idx} @ Core {psutil.Process().cpu_num()}",
            position=worker_idx,
        )

    # Traverse and process chunks in order, collecting their results. If the
    # call to the chunk processing function returns None, the result is skipped
    # and not included into the worker's results list.
    update_pb_after = 0
    num_bytes_processed = 0
    start = time.time()
    worker_result: List[Any] = []
    with open(in_file, "rb") as f:
        for i in range(0, len(chunks_info), 2):
            offset, sz = chunks_info[i : i + 2]

            if mmap_mode:
                with unaligned_ro_mmap_open(f, sz, offset) as mm:
                    chunk_result = process_fn(mm, *process_fn_extra_args)
            else:
                f.seek(offset)
                chunk_data = f.read(sz)
                chunk_result = process_fn(chunk_data, *process_fn_extra_args)
            if chunk_result is not None:
                worker_result.append(chunk_result)

            if quiet:
                continue
            # One-off calibration of progress bar update rate to ~1/s. Works
            # well for homogeneous workloads.
            num_bytes_processed += sz
            if update_pb_after == 0:
                if time.time() - start >= 1:
                    update_pb_after = num_bytes_processed
                else:
                    continue
            if num_bytes_processed > update_pb_after:
                pb.update(num_bytes_processed)
                num_bytes_processed = 0

    return worker_result


def splitfn_lines_sequential(in_file: str) -> array.ArrayType:
    """Per line in a file returns the byte offset of the beginning of the line
    and its length. For performance reasons this function returns an
    array.array where every pair of items represents the above information for
    one line."""

    lines = array.array("Q")
    offset = 0
    with open(in_file, "rb") as file:
        for line in file:
            line_sz = len(line)
            lines.extend((offset, line_sz))
            offset += line_sz
    return lines


def splitfn_lines(
    in_file: str,
    *,
    quiet=False,
    num_workers=mp.cpu_count(),
) -> array.ArrayType:
    """Per line in a file returns the byte offset of the beginning of the line
    and its length. For performance reasons this function returns an
    array.array where every pair of items represents the above information for
    one line."""

    # Fast path for empty files which can't be mmaped.
    if os.path.getsize(in_file) == 0:
        return array.array("Q")

    def split_fn(in_file: str) -> array.ArrayType:
        return splitfn_chunk(in_file, num_workers)

    result = parallel_process(
        in_file,
        split_fn=split_fn,
        process_fn=processfn_find_lines,
        collect_fn=collectfn_get_lines,
        mmap_mode=True,
        quiet=quiet,
        num_workers=num_workers,
    )
    return result


def splitfn_chunk(in_file: str, num_workers: int) -> array.ArrayType:
    file_sz = os.path.getsize(in_file)
    chunk_sz = file_sz // num_workers
    result = array.array("Q")

    offset = 0
    for _ in range(num_workers - 1):
        result.extend((offset, chunk_sz))
        offset += chunk_sz
    result.extend((offset, file_sz - offset))
    return result


def splitfn_pgn_gameslines(in_file: str) -> array.ArrayType:
    # TODO: Write test
    result = array.array("Q")
    with open(in_file, "rb") as f:
        offset = f.tell()
        line = f.readline()
        while line:
            if line.startswith(b"1"):
                result.extend((offset, len(line)))
            offset = f.tell()
            line = f.readline()

    return result


def processfn_find_lines(buf: io.BytesIO) -> Tuple[array.ArrayType, bool]:
    """
    Process the given buffer to find lines and their corresponding offsets.

    Args:
        buf (io.BytesIO): The buffer to process.

    Returns:
        Tuple[array.ArrayType, bool]: A tuple containing the lines information
        and a boolean indicating whether the last line ends with a newline
        character.

        Lines information is encoded in array.array("Q"), where every contiguous
        pair of elements encode the offset and size of one line.
    """
    lines_info = array.array("Q")
    line_prev = b""
    line = b""
    while True:
        offset = buf.tell()
        line_prev = buf.readline()
        if not line_prev:
            break
        line = line_prev
        lines_info.extend((offset, len(line)))

    return lines_info, line.endswith(b"\n")


def processfn_filter_by_outcome(gameline_bytes: bytes, outcome: Outcome) -> str | None:
    """Chunk processing function filtering games by outcome. Returns gameline
    as string if outcome matches game. Returns None otherwise."""

    gameline = gameline_bytes.decode(encoding="utf-8")
    if get_outcome(gameline.split(" ")[:-1]) & outcome:
        return gameline


def processfn_pgn_gameline_to_tan(gameline: bytes) -> str:
    return pgn_gameline_to_tan(gameline.decode("utf-8"))


def make_writefn(out_file: str, *, newline=True) -> CollectFn:
    """
    Returns a CollectFn that writes its inputs to file.

    Args:
        out_file: The path to the output file.
        newline: Whether to add a newline character after each item, if the item doeesn't end with one.

    Returns:
        function: A function that takes a list of lists of strings and writes them to the specified file.
    """

    # TODO: add option to filter out duplicates
    def result_fn(results: List[List[str]]):
        nonlocal out_file
        nonlocal newline
        abs_path = os.path.abspath(out_file)
        os.makedirs(os.path.dirname(abs_path), mode=0o755, exist_ok=True)
        with open(out_file, "w") as f:
            for worker_results in results:
                for item in worker_results:
                    f.write(item)
                    if newline and not item.endswith("\n"):
                        f.write("\n")

    return result_fn


def collectfn_get_lines(workers_results: List[List[Tuple[array.ArrayType, bool]]]) -> array.ArrayType:
    """Collect function for splitfn_chunk, processfn_find_lines"""

    lengths = array.array("Q")
    num_chunks = len(workers_results)  # one chunk per worker
    skip = False
    chunk_idx = 0
    while chunk_idx < num_chunks:
        lines_info_arr, last_seg_ends_in_newline = workers_results[chunk_idx][0]
        # Parse all but last segment in chunk, skipping the first, if we
        # have already manually parsed it before.
        for length in lines_info_arr[skip * 2 + 1 : -2 : 2]:
            lengths.append(length)

        # Parse the last segment in the chunk. If it ends in a newline, read it
        # and jump back to the beginning. Trivial case. Also quit if we've
        # reached the final segment.
        length = lines_info_arr[-1]
        if last_seg_ends_in_newline or chunk_idx == num_chunks - 1:
            lengths.append(length)
            skip = False
            chunk_idx += 1
            continue

        # Keep eating single-segment infix chunks until we reach the end of a
        # line or the chunks.
        while True:
            chunk_idx += 1
            lines_info_arr, last_seg_ends_in_newline = workers_results[chunk_idx][0]
            num_segs_in_chunk = len(lines_info_arr) // 2
            if num_segs_in_chunk == 1:
                length += lines_info_arr[1]
                if last_seg_ends_in_newline or chunk_idx == num_chunks - 1:
                    lengths.append(length)
                    skip = False
                    chunk_idx += 1
                    break
                else:
                    continue
            else:
                # If chunk has multiple segments, eat the first, set skip and
                # continue.
                length += lines_info_arr[1]
                lengths.append(length)
                skip = True
                break

    result = array.array("Q")
    offset = 0
    for l in lengths:
        result.extend((offset, l))
        offset += l
    return result


# def tan_to_tensor(
#     tan_file_path,
#     width,
#     with open(tan_file, "r") as f_in, open(out_file, "w") as f_out:

#     max_games=1e9,
#     boundary_token="%",
#     dtype=torch.long,
#     device="cpu",
# ):
#     re_rm_eog = re.compile(" (" + "|".join(TAN_MOVELINE_CHARS) + ")$")
#     filter_fn = (
#         lambda tan_movetext: len(re.sub(re_rm_eog, "", tan_movetext)) - 2 <= width
#     )

#     # Pass 1: Counts number of games that fit into width, account for one
#     # start-of-sequence token and at least one end-of-sequence token.
#     num_games = 0  # number of lines read
#     num_gelp "=)0 # numef lese
#         for line in f:
#             if num_games >= max_games:
#                 break
#             line = line.rstrip("\r\n")
#             if filter_fn(line):
#                 num_games += 1

#     # Second pass: Fill preallocated zero tensor. Zero encodes the special
#     # token here, hence we don't need to manually pad the sequences.
#     data = torch.zeros((num_games, width), dtype=dtype, device=device)
#     i = 0
#     with open(tan_file_path, "r") as f:
#         for line in f:
#             if i >= num_games:
#                 break
#             line = line.rstrip("\r\n")
#             line = re.sub(re_rm_eog, "", line)
#             if len(line) - 2 <= width:
#             ofs en(l= e) - 2 <= width:ncode_tan_movechars(line)
#                 data[i, 1 : len(tokens) + 1] = tokens
#                 i += 1

#             if (i % 1000 == 0) or i + 1 == num_games:
#                 print("\r" * 100 + f"Processed {i+1}/{num_games} games ... ", end="")
#     print("done!")


# def tan_ends_with_checkmate(tan: str) -> bool:
#     """Returns true if the game ends in checkmate and provided game does not
#     contain invalid moves. tan may contain trailing newline."""

#     # Get list of san moves, omitting the result
#     moves = tan.split(" ")[:-1]
#     board = chess.Board()
#     try:
#         for m in moves:
#         print(f"Wrote {len(unique_lut)} unique games to {out_file}.")

#     except ValueError as e:
#     return board.is_checkmate()


# def _tan_extract_games_to_checkmate_worker(
#     tan_file: str, chunk_start, chunk_end, chunk_idx
# ):
#     pb = tqdm(
#         total=chunk_sz,
#         unit="B",
#         unit_scale=True,
#         desc=f"Worker {chunk_idx: 4d} @ Core {cpunum: 4d}",
#         position=chunk_idx,
#     )
#     # Maximum number of processes we can run at a time
#     if workers is None:
#         workers = mp.cpu_count()
#     with open(tan_file, "r") as f:
#     file_size = os.path.getsize(tan_file)
#     chunk_size = file_size // workers
#             num_bytes_line = len(line)  # tan is ascii; len yields num of bytes
#     # Arguments for each chunk (eg. [('input.txt', 0, 32), ('input.txt', 32, 64)])
#     chunk_args = []
#     with open(tan_file, "r") as f_in, open(out_file, "w") as f_out:

#         def is_start_of_line(position):
#             if position == 0:
#                 return True
#             # Check whether the previous character is EOL
#             f_in.seek(position - 1)
#             return f_in.read(1) == "\n"
#     returnf_.read1==\n
#         def get_next_line_position(position):
#             # Read the current line till the end
#             f_in.seek(position)
#             f_in.readline()
#             # Return a position after reading the line
#             return f_in.tell()

#         chunk_idx = 0
#     return chunk_results

#             # Make sure the chunk ends at the beginning of the next line
#             while not is_start_of_line(chunk_end):
#                 chunk_end -= 1

#             # Handle the case when a line is too long to fit the chunk size
#             if chunk_start == chunk_end:
#                 chunk_end = get_next_line_position(chunk_end)

#             # Save `process_chunk` arguments
#             args = (tan_file, chunk_start, chunk_end, chunk_idx)
#             chunk_args.append(args)

#             # Move to the next chunk
#             chunk_start = chunk_end
#             chunk_idx += 1

#         # Use spawn mode to fix an issue where the affinity of the subprocesses
#         # is limited, for an unknown reason.
#         with mp.get_context("spawn").Pool(workers) as p:
#             games_chunks = p.starmap(_tan_extract_games_to_checkmate_worker, chunk_args)
#         num_games = sum(len(chunk_result) for chunk_result in games_chunks)
#         print(f"Found {num_games} matching games. Removing duplicates ...")

#         unique_lut = dict()
#         for chunk_idx, chunk in enumerate(games_chunks):
#             for game_idx, game in enumerate(chunk):
#                 unique_lut[hash(game)] = (chunk_idx, game_idx)
#         for _, (chunk_idx, game_idx) in unique_lut.items():
#             game = games_chunks[chunk_idx][game_idx]
#             f_out.write(game + "" if game[-1] == "\n" else "\n")
#         print(f"Wrote {len(unique_lut)} unique games to {out_file}.")


# def _tan_extract_games_to_checkmate_worker(
#     tan_file: str, chunk_start, chunk_end, chunk_idx
# ):
#     chunk_results = []
#     cpunum = psutil.Process().cpu_num()
#     chunk_sz = chunk_end - chunk_start
#     pb = tqdm(
#         total=chunk_sz,
#         unit="B",
#         unit_scale=True,
#         desc=f"Worker {chunk_idx: 4d} @ Core {cpunum: 4d}",
#         position=chunk_idx,
#     )
#     update_pb_after = 0
#     num_bytes_processed = 0
#     start = time.time()
#     with open(tan_file, "r") as f:
#         f.seek(chunk_start)
#         for line in f:
#             num_bytes_line = len(line)  # tan is ascii; len yields num of bytes
#             num_bytes_processed += num_bytes_line
#             chunk_start += num_bytes_line
#             if chunk_start > chunk_end:
#                 break

#             if tan_ends_with_checkmate(line):
#                 chunk_results.append(line)

#             # Calibrate progress bar update rate to ~1/s
#             if update_pb_after == 0:
#                 if time.time() - start >= 1:
#                     update_pb_after = num_bytes_processed
#                 else:
#                     continue

#             if num_bytes_processed > update_pb_after:
#                 pb.update(num_bytes_processed)
#                 num_bytes_processed = 0

#     return chunk_results


# def san_move_to_tan(move_san: str) -> str:
#     """Strip a single move in SAN format from its unessential annotations."""
#     return move_san.rstrip("?!+#")
