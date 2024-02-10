from itertools import product
import pytest

import chess

from src.tan_chess import (
    vs_self,
    vs_random,
    one_move_puzzle,
    RandomPlayer,
    PresetPlayer,
)
from ..assets.games import conclusive_games


class Test_vs_random:
    assert vs_random.__name__ == "vs_random"

    @pytest.mark.parametrize("num_games,play_as,num_workers", product([1, 2, 4, 8], ["white", "black"], [1, 2, 4, 8]))
    def test_random_vs_random(self, num_games, play_as, num_workers):
        p = RandomPlayer()
        games = vs_random(p, play_as, num_games, num_workers)

        assert len(games) == num_games


class Test_vs_self:
    assert vs_self.__name__ == "vs_self"

    @pytest.mark.parametrize("num_games", [1, 2, 4, 8])
    def test_random_vs_self(self, num_games):
        p = RandomPlayer()
        games = vs_self(p, num_games)

        assert len(games) == num_games

    @pytest.mark.parametrize("movelist,outcome,num_games", [(g, outcome, n) for n in range(1, 5) for outcome, games in conclusive_games.items() for g in games])
    def test_preset_vs_self(self, movelist, outcome, num_games):
        p = PresetPlayer(movelist)
        games = vs_self(p, num_games)

        assert len(games) == num_games
        assert all((g["outcome"] == outcome for g in games))


class Test_one_move_puzzle:
    assert one_move_puzzle.__name__ == "one_move_puzzle"

    @pytest.mark.parametrize("movelist,num_attempts", [(g, n) for n in range(1, 5) for games in conclusive_games.values() for g in games])
    def test_preset(self, movelist, num_attempts):
        opening_moves = movelist[:-1]
        candidate_moves = [movelist[-1]]
        board = chess.Board()
        for m in opening_moves:
            board.push_san(m)
        num_legal_moves = len(list(board.legal_moves))
        p = PresetPlayer(movelist)
        result = one_move_puzzle(p, opening_moves, candidate_moves, num_attempts=num_attempts)

        assert result["opening_moves"] == opening_moves
        assert result["candidate_moves"] == candidate_moves
        assert result["num_attempts"] == num_attempts
        assert result["num_correct"] == num_attempts
        assert result["num_legal_moves"] == num_legal_moves
