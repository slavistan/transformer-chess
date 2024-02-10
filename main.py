"""CLI interface."""

import multiprocessing as mp
import os
import time
import logging

from filelock import FileLock, Timeout
import clize
import torch

from src import db_utils, tan_chess
from src.tan_chess import GUIPlayer
from src.transformer.tokenizer import encode_tan_file
from src.transformer.transformer_player import TransformerPlayer, full_eval_transformer

logging.basicConfig(
    format="[%(filename)s:%(lineno)s@%(funcName)s][%(levelname)s] %(message)s",
    level=logging.INFO,
)


def eval_perf(
    pth_file: str,
    *,
    output_dir=None,
    num_random=16,
    num_self=16,
    num_puzzle=64,  # FIXME: Off-by-one Fehler in der Anzahl der Puzzles
    num_puzzle_attempts=16,
    num_workers=1,
    device="cpu",
):
    """
    Evaluates the performance of a transformer player.

    :param pth_file: path to VanillaTransformer checkpoint.
    :param output_dir: directory to store the evaluation results in.
    :param num_random: number of games to play against random player.
    :param num_self: number of games to play against self.
    :param num_puzzle: number of one-move checkmate puzzles to player.
    :param num_puzzle_attempts: number of attempts per puzzle.
    :param num_workers: number of cpus for multiprocessing.
    """

    # TODO: Fix CUDA error appearing after termination of the evaluation process.
    #       [W CudaIPCTypes.cpp:15] Producer process has been terminated before
    #       all shared CUDA tensors released. See Note [Sharing CUDA tensors]

    # Create a lockfile so that only one process is running concurrently. The
    # forward passes of a transformer are heavily parallized already, so that
    # running multiple evaluations would just congest the system without any
    # performance benefits, even for systems with lots of cores.
    lock = FileLock("/tmp/chess-transformer-eval.lock")
    try:
        with lock.acquire(timeout=0):
            if output_dir is None:
                output_dir = pth_file + f"-eval-{int(time.time())}"
            os.makedirs(output_dir, mode=0o755, exist_ok=True)

            output_data_path = f"{output_dir}/eval.json"
            output_pdf_path = f"{output_dir}/eval.pdf"
            stdout_path = f"{output_dir}/stdout"
            stderr_path = f"{output_dir}/stderr"

            # FIXME: Puzzle Wahrscheinlichkeit des neuen TransformerPlayers ist z.T. auf 0.0%.
            #        Wie ist das überhaupt möglich?
            result = full_eval_transformer(
                pth_file=pth_file,
                data_output_path=output_data_path,
                report_output_path=output_pdf_path,
                num_random=num_random,
                num_self=num_self,
                num_puzzles=num_puzzle,
                num_puzzle_attempts=num_puzzle_attempts,
                num_workers=num_workers,
                device=device,
            )

            with open(stdout_path, "wb") as f:
                f.write(result.stdout)
            with open(stderr_path, "wb") as f:
                f.write(result.stderr)
            print(f"Done. Output directory: '{output_dir}'")
    except Timeout:
        print("Another evaluation process is running. Abort.")


def pgn_to_tan(pgn_file: str, *, output=None):
    """Extracts gamelines from a pgn database and converts them to tan format.

    :param pgn_file: Path fo file containing games.
    :param out_file: Output file path. If left empty, '-<UNIX_TIMESTAMP>.tan' is appended to the input file path.
    """

    if output is None:
        output = pgn_file + f"-{int(time.time())}.tan"

    db_utils.pgn_to_tan_sequential(pgn_file, output)


def filter_tan(tan_file: str, outcome_union_str: str, *, output=None, num_workers=mp.cpu_count()):
    """Filters a database of chess games in TAN format by outcome.

    :param tan_file: Path to file containing newline-separated gamelines.
    :param outcome_union_str: Union string of 'tan_chess.Outcome's. E.g. 'WHITE_WINS_CHECKMATE|DRAW_STALEMATE'.
    :param out_file: Output file path. If left empty, '-filtered-<UNIX_TIMESTAMP>.tan' is appended to the input file path.
    :param num_workers: Number of workers for parallel processing. Defaults to number of available cores.
    """

    if output is None:
        output = tan_file + f"-filtered-{int(time.time())}.tan"
    if num_workers <= 0:
        num_workers = mp.cpu_count()
    outcome = tan_chess.Outcome.from_union_string(outcome_union_str)
    print(f"{tan_file}, {output}, {outcome}, {num_workers=}")

    split_fn = db_utils.splitfn_lines_sequential
    process_fn = db_utils.processfn_filter_by_outcome
    process_fn_extra_args = (outcome,)
    collect_fn = db_utils.make_writefn(str(output))
    db_utils.parallel_process(tan_file, split_fn, process_fn, collect_fn, process_fn_extra_args=process_fn_extra_args, num_workers=num_workers, quiet=False)


def play_model(
    pt_file: str,
    side="white",
):
    """
    Plays a chess game against a trained transformer model checkpoint.

    :param pt_file: Path to transformer model file.
    :param side: Side to play as, either 'white' or 'black'.
    """

    m = torch.load(pt_file).to("cpu")
    model_player = TransformerPlayer(m)
    gui_player = GUIPlayer()
    players = [gui_player, model_player]
    if side == "black":
        players = players[::-1]
    tan_chess.play_game(*players)
    # TODO: Preserve window after game ends and show result. Probably needs a
    # different setup for playing games altogehter.


def tokenize(
    tan_file_path: str,
    context_size: int,
    *,
    output_path=None,
):
    """
    Tokenizes the games in a TAN file and saves the resulting tensor of dtype
    'torch.uint8' and of width 'context_size + 1' to disk. Only games whose
    token sequences fit the width are processed.

    :param tan_file_path: Path to the input file of gamelines in TAN format.
    :param context_size: Context size of the transformer to be trained. The width of the resulting tensor will be 'context_size + 1'.
    :param output_path: Output path of tensor file. Is left empty, '_context-sz=<CONTEXT_SIZE>.pt' will be appended to the input file path to be used as output path. Subdirectories will be created.
    """

    tan_file_path = os.path.abspath(tan_file_path)
    if output_path is None:
        output_path = tan_file_path + f"_context-sz={context_size}.pt"
    else:
        output_path = os.path.abspath(output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True, mode=0o644)
    data = encode_tan_file(tan_file_path, context_size)
    torch.save(data, output_path)


def main():
    clize.run(
        {
            filter_tan.__name__: filter_tan,
            pgn_to_tan.__name__: pgn_to_tan,
            play_model.__name__: play_model,
            eval_perf.__name__: eval_perf,
            tokenize.__name__: tokenize,
        }
    )


if __name__ == "__main__":
    main()
