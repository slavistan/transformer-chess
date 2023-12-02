"""CLI interface."""

import multiprocessing as mp
import resource
import os
import time
import sys
import logging
from torch import cuda
import clize
from src import db_utils, san_chess, vanilla_transformer, performance

logging.basicConfig(
    format="[%(filename)s:%(lineno)s@%(funcName)s][%(levelname)s] %(message)s",
    level=logging.INFO,
)

def eval(
    pth_file: str,
    *,
    output_dir=None,
    num_random=16,
    num_self=16,
    num_puzzle=64,
    num_puzzle_attempts=16,
    num_tries=8, # tries to generate a valid move (transformer player)
    num_workers=1,
):
    """
    Evaluates the performance of a transformer player.

    :param pth_file: Path to vanilla_transformer.Model checkpoint.
    :param output_data: Where to store the evaluation metrics in json format.
    :param output_pdf: Where to store the generated report in pdf format.
    :num_random: number of games to player against random player.
    :num_self: number of games to player against self.
    :num_puzzle: number of one-move checkmate puzzles to player.
    :num_puzzle_attempts: number of attempts per puzzle.
    :num_tries: number of tries to generate a valid move (relevant for transformer players).
    :num_workers: number of cpus for multiprocessing.
    """

    if output_dir is None:
        output_dir = pth_file + f"-eval-{int(time.time())}"
    os.makedirs(output_dir, mode=0o755, exist_ok=True)

    output_data_path = f"{output_dir}/eval.json"
    output_pdf_path = f"{output_dir}/eval.pdf"
    stdout_path = f"{output_dir}/stdout"
    stderr_path = f"{output_dir}/stderr"

    result = performance.full_eval_transformer(
        pth_file=pth_file,
        data_output_path=output_data_path,
        report_output_path=output_pdf_path,
        num_random=num_random,
        num_self=num_self,
        num_puzzles=num_puzzle,
        num_puzzle_attempts=num_puzzle_attempts,
        num_tries=num_tries,
        num_workers=num_workers,
    )

    with open(stdout_path, "wb") as f:
        f.write(result.stdout)
    with open(stderr_path, "wb") as f:
        f.write(result.stderr)
    print(f"Done. Output directory: '{output_dir}'")








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
    :param outcome_union_str: Union string of 'san_chess.Outcome's. E.g. 'WHITE_WINS_CHECKMATE|DRAW_STALEMATE'.
    :param out_file: Output file path. If left empty, '-filtered-<UNIX_TIMESTAMP>.tan' is appended to the input file path.
    :param num_workers: Number of workers for parallel processing. Defaults to number of available cores.
    """

    if output is None:
        output = tan_file + f"-filtered-{int(time.time())}.tan"
    if num_workers <= 0:
        num_workers = mp.cpu_count()
    outcome = san_chess.Outcome.from_union_string(outcome_union_str)
    print(f"{tan_file}, {output}, {outcome}, {num_workers=}")

    split_fn = db_utils.splitfn_lines_sequential
    process_fn = db_utils.processfn_filter_by_outcome
    process_fn_extra_args = (outcome,)
    collect_fn = db_utils.make_writefn(str(output))
    db_utils.parallel_process(tan_file, split_fn, process_fn, collect_fn, process_fn_extra_args=process_fn_extra_args, num_workers=num_workers, quiet=False)



def play_model(
    pth_file: str,
    *,
    device="cuda" if cuda.is_available() else "cpu",
    side="white",
    num_retries=8,
):
    """Plays a chess game against a model.

    :param pth_file: Path to model file.
    :param device: Device to run model on.
    :param side: Side to play as, either 'white' or 'black'.
    :param num_retries: Number of retries to allow the transformer for each move.
    """

    m = vanilla_transformer.Model.load(pth_file).to(device)
    model_player = vanilla_transformer.TransformerPlayer(m)
    gui_player = san_chess.GUIPlayer()
    players = [gui_player, model_player]
    retries = (0, num_retries)
    if side == "black":
        players = players[::-1]
        retries = retries[::-1]
    san_chess.play_game(*players, num_retries=retries)
    # TODO: Preserve window after game ends and show result. Probably needs a
    # different setup for playing games altogehter.


def main():
    clize.run(
        {
            filter_tan.__name__: filter_tan,
            pgn_to_tan.__name__: pgn_to_tan,
            play_model.__name__: play_model,
            eval.__name__: eval,
        }
    )


if __name__ == "__main__":
    if os.environ.get("MAX_RAM", None) is not None:
        # TODO: Doesn't work; probably doesn't really measure the right ram usage
        max_ram = int(float(os.environ["MAX_RAM"]))
        logging.info(f"Limit max RAM usage of process to {max_ram} bytes.")
        resource.setrlimit(resource.RLIMIT_AS, (max_ram, max_ram))
        try:
            main()
        except MemoryError as e:
            logging.error(f"Process terminated due to exceeding the set memory limit of {max_ram} bytes.")
            sys.exit(1)
    else:
        main()
