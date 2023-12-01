from typing import Tuple, Callable, TypedDict, Iterable
from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass
import multiprocessing as mp
from itertools import repeat
import logging
import numpy as np
import time
import pandas as pd
import chess
from src.san_chess import SANPlayer, RandomPlayer, Game, play_game, Literal, List
from src.db_utils import tan_gameline_to_moveline, san_move_to_tan
from plotnine import *


class PuzzleResult(TypedDict):
    opening_moves: List[str]
    candidate_moves: List[str]
    num_legal_moves: int
    num_attempts: int
    num_correct: int


def _vs_random(
    player: SANPlayer,
    num_games: int,
    *,
    play_as: Literal["white", "black"] = "white",
    num_retries: int = 0,
    retry_strategy: Literal["eager", "lazy"] = "lazy",
) -> List[Game]:
    results: List[Game] = []
    random_player = RandomPlayer()
    if play_as == "white":
        players = (player, random_player)
    else:
        players = (random_player, player)

    for _ in range(num_games):
        player.reset()
        random_player.reset()
        game_result = play_game(*players, num_retries=num_retries, retry_strategy=retry_strategy)
        results.append(game_result)

    return results


def vs_self(
    player: SANPlayer,
    num_games: int,
    *,
    num_retries: int = 0,
    retry_strategy: Literal["eager", "lazy"] = "lazy",
) -> List[Game]:
    results: List[Game] = []
    for _ in range(num_games):
        player.reset()
        game_result = play_game(player, num_retries=num_retries, retry_strategy=retry_strategy)
        results.append(game_result)
    return results


# TODO: Test
def one_move_puzzle(
    player: SANPlayer,
    movelist: List[str],
    candidate_moves: List[str],
    *,
    num_attempts: int = 1,  # rerun puzzle, relevant for stochastic players
    num_tries: int = 1,  # number of tries until a valid move is returned, relevant for transformer players
) -> PuzzleResult:
    player.reset(movelist)
    counter = 0

    board = chess.Board()
    for move in movelist:
        board.push_san(move)
    num_legal = len(list(board.legal_moves))

    for _ in range(num_attempts):
        move_suggestion = ""
        suggested_legal_move = False
        for _ in range(num_tries):
            sig, suggestions = player.suggest_moves(1)
            if sig is not None:
                continue
            move_suggestion = san_move_to_tan(suggestions[0])

            # Check if move is legal. Pychess doesn't offer a method to check
            # the validity of a move in SAN notation, so we have to call
            # push_san() directly and look for exceptions. However, we must not
            # modify the move stack and thus keep a buffer of the board.
            board_buf = deepcopy(board)
            try:
                board.push_san(move_suggestion)
            except ValueError:
                continue
            board = board_buf
            suggested_legal_move = True
            break

        if suggested_legal_move and move_suggestion in candidate_moves:
            counter += 1

    return {
        "opening_moves": movelist,
        "candidate_moves": candidate_moves,
        "num_legal_moves": num_legal,
        "num_attempts": num_attempts,
        "num_correct": counter,
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
            moveline = tan_gameline_to_moveline(line)
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

def uber_eval(
    player: SANPlayer,
    puzzles: List[Tuple[List[str], List[str]]],
    *,
    num_random=16,
    num_self=16,
    num_tries=8,  # number of attempts to generate a valid move (transformer players)
    num_puzzle_attempts=64,  # number of times a puzzle is tried (stochastic players)
    num_workers=1,
) -> EvalResult:
    result: EvalResult = {
        "vs_random_as_white": [],
        "vs_random_as_black": [],
        "vs_self": [],
        "one_move_checkmate_puzzles": []
    }

    # Games against random player
    for side in ["white", "black"]:
        start = time.time()
        logging.info(f"Playing {num_random} games as {side} against random player ... ")
        result[f"vs_random_as_{side}"] = vs_random(
            player,
            num_games=num_random,
            num_retries=num_tries - 1,
            play_as=side,
            num_workers=num_workers,
        )
        logging.info(f"Done after {int(time.time() - start)}s.")

    # Games against self
    logging.info(f"Playing {num_self} games against self ... ")
    start = time.time()
    result["vs_self"] = vs_self(
        player,
        num_games=num_self,
        num_retries=num_tries - 1,
    )
    logging.info(f"Done after {int(time.time() - start)}s.")

    # One-Move Puzzles
    # TODO: Anzahl legaler Züge, Länge der Züge, Anzahl Erfolge, Anzahl versuche
    logging.info(f"Playing {len(puzzles)} puzzles ... ")
    start = time.time()
    for i, (movelist, candidate_moves) in enumerate(puzzles):
        start = time.time()
        logging.info(f"{i+1}/{len(puzzles)} puzzles done ...")
        puzzle_result = one_move_puzzle(
            player,
            movelist,
            candidate_moves,
            num_attempts=num_puzzle_attempts,
        )
        result["one_move_checkmate_puzzles"].append(puzzle_result)
    logging.info(f"Done after {int(time.time() - start)}s.")

    return result


def vs_random(
    player: SANPlayer,
    num_games: int,
    *,
    play_as: Literal["white", "black"] = "white",
    num_retries: int = 0,
    retry_strategy: Literal["eager", "lazy"] = "lazy",
    num_workers: int = 1,
) -> List[Game]:
    """
    Note that there's no point in using massive parallelization if evaluating a
    transformer player on the cpu, as its forward pass is already parallelized
    by the pytorch tensor arithmetic implementation. In that case, try
    `num_workers` of 2 or 3 if you have a lot of cores.
    """

    num_games_per_worker = num_games // num_workers
    worker_pargs = num_workers * [
        [
            player,
            num_games_per_worker,
        ]
    ]
    worker_kwargs = num_workers * [{"play_as": play_as, "num_retries": num_retries, "retry_strategy": retry_strategy}]
    for i in range(num_games - num_games_per_worker * num_workers):
        worker_pargs[i][1] += 1

    with mp.get_context("spawn").Pool(num_workers) as p:
        worker_results = starmap_with_kwargs(p, _vs_random, worker_pargs, worker_kwargs)

    result = sum(worker_results, [])
    return result


def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(_apply_args_and_kwargs, args_for_starmap)


def _apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)


def plot_outcome_hist(games: List[Game]) -> ggplot:
    outcomes = [str(g["outcome"]).partition(".")[2] for g in games]
    df = pd.DataFrame.from_dict({"outcomes": outcomes})
    plot = ggplot(df) + geom_bar(aes(x="outcomes")) + theme(axis_text_x=element_text(angle=45, vjust=1.0, hjust=1))
    return plot


def plot_game_len_hist(
    games: List[Game],
    *,
    binwidth=7,
) -> ggplot:
    length = [len(g["moves"]) for g in games]
    outcome = [str(g["outcome"]).partition(".")[2] for g in games]
    df = pd.DataFrame.from_dict({"length": length, "outcome": outcome})
    plot = ggplot(df) + geom_histogram(aes(x="length", fill="outcome"), binwidth=binwidth, color="none") + labs(title="Length of games", x="Number of Moves", y="Count")
    return plot


def mean_game_len(games: List[Game]) -> Tuple[float, float]:
    lengths = [len(g["moves"]) for g in games]
    mean, std = np.mean(lengths).astype(float), np.std(lengths).astype(float)
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


# TODO: num_retries off-by-one Verhalten angleichen. Parameter sollte Anzahl der Iterationen angeben.
