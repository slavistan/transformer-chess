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
    output=None,
    num_random=16,
    num_self=16,
    num_puzzle=64
):
    if output is None:
        # TODO: report-001, report-002 ... Namenskonflikte abfangen
        output = pth_file + "-report.pdf"

    m = vanilla_transformer.Model.load(pth_file).to("cpu")
    p = vanilla_transformer.TransformerPlayer(m)

    result = {}

    # Games against random player
    for side in ["white", "black"]:
        start = time.time()
        logging.info(f"Playing {num_random} games against random player as {side} ... ")
        result[f"vs-random-as-{side}"] = performance.vs_random(
            p,
            num_games=num_random,
            num_retries=8,
            play_as=side,
            num_workers=2,
        )
        logging.info(f"Done after {int(time.time() - start)}s.")

    # Games against self
    logging.info(f"Playing {num_self} games against self ... ")
    result["vs-self"] = performance.vs_self(
        p,
        num_games=num_self,
        num_retries=8
    )







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
