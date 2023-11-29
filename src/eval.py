from typing import Tuple, Callable
from src.san_chess import SANPlayer, RandomPlayer, Game, play_game, Literal, List


# Mate in one Puzzles

def vs_random(
    player: SANPlayer,
    num_games: int,
    *,
    play_as: Literal["white", "black"] = "white",
    num_retries: int = 0,
    retry_strategy: Literal["eager", "lazy"] = "lazy"
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
    retry_strategy: Literal["eager", "lazy"] = "lazy"
) -> List[Game]:

    results: List[Game] = []
    for _ in range(num_games):
        player.reset()
        game_result = play_game(player, num_retries=num_retries, retry_strategy=retry_strategy)
        results.append(game_result)
    return results



# Evaluatio
# fight random bob
# - result histogram
# - average game length
# - num of retries

