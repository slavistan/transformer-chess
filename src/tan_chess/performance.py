from typing import Tuple, List, Literal, TypedDict, Iterable, cast
import subprocess
import time
import os
import multiprocessing as mp

import numpy as np
import pandas as pd
import chess
from plotnine import *

from src.tools import starmap_with_kwargs
from .common import (
    TANMove,
    TANMoveList,
    TANPlayer,
    tan_moveline_from_gameline,
    is_valid_move,
)
from .game import Game, play_game
from .players import RandomPlayer


def vs_random(
    player: TANPlayer,
    num_games: int,
    *,
    play_as: Literal["white", "black"] = "white",
    num_workers: int = 1,
) -> List[Game]:
    """
    Plays games vs a RandomPlayer.

    Note that there's no point in using massive parallelization if evaluating a
    transformer player on the cpu, as its forward pass is already parallelized
    by the pytorch tensor arithmetic implementation. In that case, try
    `num_workers` of 2 or 3 if you have a lot of cores.
    """

    # Skip multiprocessing if we're only using a single worker.
    if num_workers == 1:
        return _vs_random_worker(player, num_games, play_as=play_as)

    num_workers = min(num_workers, num_games)
    num_games_per_worker = num_games // num_workers
    worker_pargs = num_workers * [
        [
            player,
            num_games_per_worker,
        ]
    ]
    worker_kwargs = num_workers * [{"play_as": play_as}]
    for i in range(num_games - num_games_per_worker * num_workers):
        worker_pargs[i][1] += 1

    with mp.get_context("spawn").Pool(num_workers) as p:
        worker_results = starmap_with_kwargs(p, _vs_random_worker, worker_pargs, worker_kwargs)

    result = sum(worker_results, [])
    return result


def _vs_random_worker(
    player: TANPlayer,
    num_games: int,
    *,
    play_as: Literal["white", "black"] = "white",
) -> List[Game]:
    """
    Multiprocessing worker for `vs_random`.
    """

    results: List[Game] = []
    random_player = RandomPlayer()
    if play_as == "white":
        players = (player, random_player)
    else:
        players = (random_player, player)

    for _ in range(num_games):
        player.reset()
        random_player.reset()
        game_result = play_game(*players)
        results.append(game_result)

    return results


def vs_self(
    player: TANPlayer,
    num_games: int,
) -> List[Game]:
    results: List[Game] = []
    for _ in range(num_games):
        player.reset()
        game_result = play_game(player)
        results.append(game_result)
    return results


class PuzzleResult(TypedDict):
    opening_moves: TANMoveList
    candidate_moves: Iterable[TANMove]
    num_legal_moves: int
    num_attempts: int
    num_correct: int


def one_move_puzzle(
    player: TANPlayer,
    opening_moves: TANMoveList,
    candidate_moves: Iterable[TANMove],
    *,
    num_attempts: int = 1,  # rerun puzzle, relevant for stochastic players
) -> PuzzleResult:
    player.reset()
    player.push_moves(opening_moves)

    board = chess.Board()
    for move in opening_moves:
        board.push_san(move)

    num_correct = 0
    for _ in range(num_attempts):
        move = player.suggest_move()
        if not isinstance(move, TANMove):
            continue

        if not is_valid_move(move, board):
            continue

        if move in candidate_moves:
            num_correct += 1

    return {
        "opening_moves": opening_moves,
        "candidate_moves": candidate_moves,
        "num_legal_moves": len(list(board.legal_moves)),
        "num_attempts": num_attempts,
        "num_correct": num_correct,
    }


def one_move_puzzle_from_tan(tan_file: str, *, num_games: int):
    """
    Splits up tan gamelines read from a file into the puzzle movelist and
    candidate moves to be used by one_move_puzzle. Used to make puzzles from
    games that end in checkmate.

    Returns a generator yielding tuples of (movelist, candidate_moves).
    """

    with open(tan_file, "r") as f:
        for i, line in enumerate(f):
            moveline = tan_moveline_from_gameline(line)
            movelist = moveline.split(" ")
            puzzle_movelist = movelist[:-1]
            puzzle_candidate_moves = [movelist[-1]]
            yield puzzle_movelist, puzzle_candidate_moves

            if i >= num_games:
                break


class EvalResult(TypedDict):
    vs_random_as_white: List[Game]
    vs_random_as_black: List[Game]
    vs_self: List[Game]
    one_move_checkmate_puzzles: List[PuzzleResult]


def full_eval(
    player: TANPlayer,
    puzzles: List[Tuple[List[str], List[str]]],
    *,
    num_random=16,
    num_self=16,
    num_puzzle_attempts=64,  # number of times a puzzle is tried (stochastic players)
    num_workers=1,
) -> EvalResult:
    result: EvalResult = {
        "vs_random_as_white": [],
        "vs_random_as_black": [],
        "vs_self": [],
        "one_move_checkmate_puzzles": [],
    }

    # Games against random player
    for side in ("white", "black"):
        start = time.time()
        print(f"Playing {num_random} games as {side} against random player ... ")
        k = cast(Literal["vs_random_as_white", "vs_random_as_black"], f"vs_random_as_{side}")
        result[k] = vs_random(
            player,
            num_games=num_random,
            play_as=side,
            num_workers=num_workers,
        )
        print(f"Done after {int(time.time() - start)}s.")

    # Games against self
    print(f"Playing {num_self} games against self ... ")
    start = time.time()
    result["vs_self"] = vs_self(player, num_self)
    print(f"Done after {int(time.time() - start)}s.")

    # One-Move Puzzles
    # TODO: Anzahl legaler Züge, Länge der Züge, Anzahl Erfolge, Anzahl versuche
    print(f"Playing {len(puzzles)} puzzles ... ")
    start = time.time()
    for i, (movelist, candidate_moves) in enumerate(puzzles):
        start = time.time()
        print(f"{i+1}/{len(puzzles)} puzzles done ...")
        puzzle_result = one_move_puzzle(
            player,
            movelist,
            candidate_moves,
            num_attempts=num_puzzle_attempts,
        )
        result["one_move_checkmate_puzzles"].append(puzzle_result)
    print(f"Done after {int(time.time() - start)}s.")

    return result


def plot_outcome_hist(games: List[Game]) -> ggplot:
    outcomes = [str(g["outcome"]).partition(".")[2] for g in games]
    df = pd.DataFrame.from_dict({"outcomes": outcomes})
    plot = ggplot(df) + geom_bar(aes(x="outcomes")) + theme(axis_text_x=element_text(angle=45, vjust=1.0, hjust=1))
    return plot


def plot_game_len_hist(
    games: Iterable[Game],
    *,
    binwidth=7,
) -> ggplot:
    length = [len(g["moves"]) for g in games]
    outcome = [str(g["outcome"]).partition(".")[2] for g in games]
    df = pd.DataFrame.from_dict({"length": length, "outcome": outcome})
    plot = ggplot(df) + geom_histogram(aes(x="length", fill="outcome"), binwidth=binwidth, color="none") + labs(title="Length of games", x="Number of Moves", y="Count")
    return plot


def mean_game_len(games: Iterable[Game]) -> Tuple[float, float]:
    lengths = [len(g["moves"]) for g in games]
    mean, std = np.mean(lengths), np.std(lengths)
    return mean, std


def plot_puzzle_likelihood_vs_num_of_legal_moves(puzzle_results: Iterable[PuzzleResult]) -> ggplot:
    puzzles = list(puzzle_results)
    df = pd.DataFrame.from_dict({"num_legal_moves": [p["num_legal_moves"] for p in puzzles], "p_correct": [p["num_correct"] / p["num_attempts"] for p in puzzles]})
    plot = (
        ggplot(df)
        + geom_point(
            aes(x="num_legal_moves", y="p_correct"),
            alpha=0.25,
        )
        + labs(title="Number of legal moves vs likelihood", x="Number of legal moves", y="Likelihood")
    )
    return plot


def plot_puzzle_likelihood_vs_len_of_opening(puzzle_results: Iterable[PuzzleResult]) -> ggplot:
    puzzles = list(puzzle_results)
    df = pd.DataFrame.from_dict({"len_opening": [len(p["opening_moves"]) for p in puzzles], "p_correct": [p["num_correct"] / p["num_attempts"] for p in puzzles]})
    plot = (
        ggplot(df)
        + geom_point(
            aes(x="len_opening", y="p_correct"),
            alpha=0.25,
        )
        + labs(title="Length of Opening Sequence vs Likelihood", x="Length of Opening Sequence", y="Likelihood")
    )
    return plot


def plot_puzzle_len_of_opening_vs_num_legal_moves(puzzle_results: Iterable[PuzzleResult]) -> ggplot:
    puzzles = list(puzzle_results)
    df = pd.DataFrame.from_dict({"len_opening": [len(p["opening_moves"]) for p in puzzles], "num_legal_moves": [p["num_legal_moves"] for p in puzzles]})
    plot = (
        ggplot(df)
        + geom_point(
            aes(x="len_opening", y="num_legal_moves"),
            position=position_jitter(height=0.00, width=0),
            alpha=0.25,
        )
        + labs(title="Length of Opening Sequence vs Number of Legal Moves", x="Length of Opening Sequence", y="Number of Legal Moves")
    )
    return plot


def make_report(eval_json: str, output: str):
    """
    Compiles codebraid markdown.
    """

    CODEBRAID_TEMPLATE = "./eval/full-eval-codebraid.md"
    DATA_FILE_ENVVAR = "DATA_JSON"  # Envvar containing path to data file.

    # Set PYTHONPATH to current working directory, as corebraid will create and
    # switch to a temporary directory, losing the option to import out local
    # './src' modules.
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + "" if env.get("PYTHONPATH") is None else ":" + env["PYTHONPATH"]
    env[DATA_FILE_ENVVAR] = eval_json

    result = subprocess.run(
        [
            "codebraid",
            "pandoc",
            "--no-cache",
            "--overwrite",
            "--from",
            "markdown",
            "--to",
            "pdf",
            "-o",
            output,
            CODEBRAID_TEMPLATE,
        ],
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result


# def full_eval_transformer(
#     pth_file: str,
#     data_output_path: str,
#     report_output_path: str,
#     *,
#     num_random=64,
#     num_self=64,
#     num_puzzles=256,
#     num_puzzle_attempts=64,
#     num_workers=1,
#     num_tries_until_valid=16,
# ):
#     # TODO: machine-independent way of storing puzzles
#     puzzles = list(
#         one_move_puzzle_from_tan(
#             "./data/2309-checkmate.tan",
#             num_games=num_puzzles,
#         )
#     )

#     m = vanilla_transformer.Model.load(pth_file).to("cpu")
#     player = vanilla_transformer.TransformerPlayer(m, num_tries_until_valid=num_tries_until_valid)
#     eval_results = full_eval(
#         player,
#         puzzles,
#         num_random=num_random,
#         num_self=num_self,
#         num_puzzle_attempts=num_puzzle_attempts,
#         num_workers=num_workers,
#     )

#     with open(data_output_path, "w") as f:
#         json.dump(eval_results, f, indent=4, default=str)

#     return make_report(data_output_path, report_output_path)



# TODO: Benchmark-Ideen:
#       - Länge der Spiele gegen {sich selbst,RandomPlayer}, die aufgrund Zugunfähigkeit enden, als Histogramm anzeigen