from src.san_chess import SANPlayer, RandomPlayer
from typing import Tuple


# fight random bob
# - result histogram
# - average game length
# - num of retries

# self play

# Mate in one Puzzles

def vs_random(
    player: SANPlayer,
    num_games: int | Tuple[int, int],
    *,
    num_retries: int = 0
):
    if isinstance(num_games, int):
        as_white = num_games // 2
        as_black = num_games - as_white
        num_games = (as_white, as_black)


