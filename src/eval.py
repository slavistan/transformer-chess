from typing import Tuple, Callable
from src.san_chess import SANPlayer, RandomPlayer, Game, play_game, Literal, List


# fight random bob
# - result histogram
# - average game length
# - num of retries

# self play

# Mate in one Puzzles

def vs_random(
    player_factory: Callable[[], SANPlayer], # TODO: implement reset() fn for SAN Player
    num_games: int | Tuple[int, int],
    *,
    num_retries: int = 0,
    retry_strategy: Literal["eager", "lazy"] = "lazy"
):
    if isinstance(num_games, int):
        as_white = num_games // 2
        as_black = num_games - as_white
    else:
        as_white = num_games[0]
        as_black = num_games[1]

    game_results: List[Game] =[]
    for _ in range(as_white):
        p1 = player_factory()
        p2 = RandomPlayer()
        game_result = play_game(p1, p2, num_retries=num_retries, retry_strategy=retry_strategy)
        game_results.append(game_result)

    for _ in range(as_black):
        p1 = player_factory()
        p2 = RandomPlayer()
        game_result = play_game(p2, p1, num_retries=num_retries, retry_strategy=retry_strategy)
        game_results.append(game_result)

    return game_results




