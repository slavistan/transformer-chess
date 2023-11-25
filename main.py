"""CLI interface."""

from typing import Literal

import torch
import multiprocessing as mp
import time

import clize

from src import db_utils, san_chess, vanilla_transformer


# TODO: Implement; Create splitfn, processfn, and collectfn
def pgn_to_tan(pgn_file: str, *, out_file=None, num_workers=None):
    """Extracts gamelines from a pgn database and converts them to tan format.

    :param pgn_file: Path fo file containing games.
    :param out_file: Output file path. If left empty, '-<UNIX_TIMESTAMP>.tan' is appended to the input file path.
    :param num_workers: Number of workers for parallel processing. Defaults to number of available cores.
    """
    # TODO: gogogogo


def filter_tan(tan_file: str, outcome_union_str: str, *, out_file=None, num_workers=mp.cpu_count()):
    """Filters a database of chess games in TAN format by outcome.

    :param tan_file: Path to file containing newline-separated gamelines.
    :param outcome_union_str: Union string of 'san_chess.Outcome's. E.g. 'WHITE_WINS_CHECKMATE|DRAW_STALEMATE'.
    :param out_file: Output file path. If left empty, '-filtered-<UNIX_TIMESTAMP>.tan' is appended to the input file path.
    :param num_workers: Number of workers for parallel processing. Defaults to number of available cores.
    """

    if out_file is None:
        out_file = tan_file + f"-filtered-{int(time.time())}.tan"
    if num_workers <= 0:
        num_workers = mp.cpu_count()
    outcome = san_chess.Outcome.from_union_string(outcome_union_str)
    print(f"{tan_file}, {out_file}, {outcome}, {num_workers=}")

    split_fn = db_utils.splitfn_lines_sequential
    process_fn = db_utils.processfn_filter_by_outcome
    process_fn_extra_args = (outcome,)
    collect_fn = db_utils.make_writefn(str(out_file))
    db_utils.parallel_process(tan_file, split_fn, process_fn, collect_fn, process_fn_extra_args=process_fn_extra_args, num_workers=num_workers, quiet=False)


def play_model(
    pth_file: str,
    *,
    device = "cuda" if torch.cuda.is_available() else "cpu",
    side = "white",
    num_retries = 8,
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
    # TODO: Preserve window after end if game and show result. Probably needs a
    # different setup for playing games altogehter.


if __name__ == "__main__":
    clize.run(
        {
            filter_tan.__name__: filter_tan,
            pgn_to_tan.__name__: pgn_to_tan,
            play_model.__name__: play_model,
        }
    )
