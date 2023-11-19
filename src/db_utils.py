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
import psutil
from typing import List, Dict, Tuple, Callable, Any, Collection, BinaryIO, Sequence
from tqdm.auto import tqdm
from src.san_chess import Outcome, get_outcome, SAN_ANNOTATION_POSTFIX

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
    re.compile(f"[{SAN_ANNOTATION_POSTFIX}]")
]

_san_movetext_eog_to_tan = {
    "1-0": "W",
    "0-1": "S",
    "1/2-1/2": "U",
    " *": "" # incomplete games are denoted by an asterisk
}

def pgn_gameline_to_tan(san_gameline: str) -> str:
    """Converts a game's PGN's movetext (aka. gameline) to trimmed algebraic
    notation. Expects a string representing a full game's movetext in algebraic
    notation. Trailing newlines will be stripped."""

    san_gameline = san_gameline.rstrip()
    for rgx in _san_movetext_re:
        san_gameline = re.sub(rgx, "", san_gameline)

    for san_eog, tan_eog in _san_movetext_eog_to_tan.items():
        if san_gameline.endswith(san_eog):
            san_gameline = san_gameline[:-len(san_eog)] + tan_eog
            break

    return san_gameline


def tan_gamelines_from_pgn(line: str):
    pass


def parallel_process(
    in_file: str,
    split_fn: Callable[[str], Sequence[int]],
    process_fn: Callable[[bytes, ...], Any],
    collect_fn: Callable[[List[List[Any]]], Any],
    *,
    process_fn_extra_args: Tuple = (),
    num_workers: int=0,
    quiet=False
):
    if num_workers == 0:
        num_workers = mp.cpu_count()

    # Identify chunks to be processed. A chunk is identified by its byte offset
    # and its size in bytes. For better data locality chunks should be sorted
    # by their offsets. Chunks don't have to be contiguous or non-overlapping.
    #
    # Each pair of contiguous elements encodes one chunk. E.g.
    #   [0, 14, 14, 7, 23, 4, 19, 4]
    #    ^  ^^
    #    ^  ^^~~~ size in bytes of chunk 0
    #    ^~~~~~~~ offset in bytes of chunk 0
    chunks_info = split_fn(in_file)
    num_chunks = len(chunks_info) // 2

    # Construct parameters for worker function, made up of
    chunks_per_worker = math.ceil(num_chunks / num_workers)
    worker_args: List[Tuple[str, Sequence[int], Callable[[bytes], Any], int]] = []
    for i in range(num_workers):
        worker_chunks_info = chunks_info[i*chunks_per_worker*2:i*chunks_per_worker*2+chunks_per_worker*2]
        worker_args.append((
            in_file,               # file path
            worker_chunks_info,    # chunks to be processed by the worker
            process_fn,            # chunk processing function
            i,                     # worker index
            process_fn_extra_args, # additional arguments for processfn
            quiet,                 # verbosity flag
        ))

    # Start parallel processing and collect the results.
    with mp.get_context("spawn").Pool(num_workers) as p:
        worker_results: List[List[Any]] = p.starmap(worker, worker_args)

    # Process the results.
    collect_fn(worker_results)


def worker(
    in_file: str,
    chunks_info: Sequence[int],
    process_fn: Callable[[bytes, ...], Any],
    worker_idx: int,
    process_fn_extra_args: Tuple[Any],
    quiet: bool
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
            offset, sz = chunks_info[i:i+2]

            # TODO: Parameterize chunks as bytes or mmep'd io
            #       Add parameter chunk_mmap: bool = False
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


def splitfn_lines(in_file: str) -> array.ArrayType:
    """Per line in a file returns the byte offset of the beginning of the line
    and its length. For performance reasons this function returns an
    array.array where every pair of items represents the above information for
    one line."""

    lines = array.array("Q")
    offset = 0
    with open(in_file, 'rb') as file:
        for line in file:
            line_sz = len(line)
            lines.extend((offset, line_sz))
            offset += line_sz
    return lines


def processfn_filter_by_outcome(gameline_bytes: bytes, outcome: Outcome) -> str | None:
    """Chunk processing function filtering games by outcome. Returns gameline
    as string if outcome matches game. Returns None otherwise."""

    gameline = gameline_bytes.decode(encoding="utf-8")
    if get_outcome(gameline.split(" ")[:-1]) & outcome:
        return gameline


def make_collectfn_write(out_file: str, *, newline=True):
    """
    Returns a function that writes a list of lists of strings to a file.

    Args:
        out_file: The path to the output file.
        newline: Whether to add a newline character after each item, if the item doeesn't end with one.

    Returns:
        function: A function that takes a list of lists of strings and writes them to the specified file.
    """

    # TODO: add option to filter out duplicates
    def result_fn(results: List[List[str]]):
        os.makedirs(os.path.dirname(out_file), mode=0o755, exist_ok=True)
        with open(out_file, "w") as f:
            for worker_results in results:
                for item in worker_results:
                    f.write(item)
                    if newline and not item.endswith("\n"):
                        f.write("\n")
    return result_fn



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
