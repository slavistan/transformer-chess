"""Tests for san_chess.py"""

import random
from itertools import product
from typing import cast, Any

import pytest
import chess

from src import san_chess
from src.san_chess import outcome_from_signal, PlayerSignal, PresetPlayer, RandomPlayer, play_game



class Test_PresetPlayer:
    assert PresetPlayer.__name__ == "PresetPlayer"

    # TODO: use parametrize
    def test_default(self):
        for _, games in san_chess.conclusive_games.items():
            for movelist in games:
                p = san_chess.PresetPlayer(movelist)
                for move in movelist:
                    response = p.suggest_move()
                    assert response == move
                    p.push_moves([move])

                # Sends out of moves signal once the movelist is exceeded.
                sig = p.suggest_move()
                assert sig == san_chess.PlayerSignal.DIVERGED_FROM_PRESET


class Test_RandomPlayer:
    assert RandomPlayer.__name__ == "RandomPlayer"

    # TODO: Test out of moves signal
    # TODO: Test basic functionality (one move positions)

class Test_play_game:
    assert play_game.__name__ == "play_game"

    def test_default(self):
        """Tests the play_game() function."""

        for outcome, games in san_chess.conclusive_games.items():
            for movelist in games:
                for i in range(min(len(movelist), 16)):
                    opening = movelist[:i]  # try prefixes as openings
                    game_1p = san_chess.play_game(san_chess.PresetPlayer(movelist), opening_moves=opening)
                    game_2p = san_chess.play_game(san_chess.PresetPlayer(movelist), san_chess.PresetPlayer(movelist), opening_moves=opening)
                    assert outcome == game_1p["outcome"]  # outcomes match
                    assert outcome == game_2p["outcome"]  # outcomes match

            for movelist in games:
                game_1p = san_chess.play_game(san_chess.PresetPlayer([]), opening_moves=movelist)
                game_2p = san_chess.play_game(san_chess.PresetPlayer([]), san_chess.PresetPlayer([]), opening_moves=movelist)
                assert outcome == game_1p["outcome"]  # outcomes match
                assert outcome == game_2p["outcome"]  # outcomes match

def test_get_outcome():

    fools_mate = san_chess.conclusive_games[san_chess.Outcome.WHITE_WINS_CHECKMATE][0]
    outcome = san_chess.get_outcome(fools_mate)
    assert outcome == san_chess.Outcome.WHITE_WINS_CHECKMATE
    outcome = san_chess.get_outcome(fools_mate[:-1])
    assert outcome == san_chess.Outcome.BLACK_WINS_RESIGNATION_DIVERGED_FROM_PRESET
    outcome = san_chess.get_outcome(fools_mate[:-2])
    assert outcome == san_chess.Outcome.WHITE_WINS_RESIGNATION_DIVERGED_FROM_PRESET

    invalid_moves = "e9 y5 f4 g5 Qh5".split(" ")
    outcome = san_chess.get_outcome(invalid_moves)
    assert outcome == san_chess.Outcome.ABORT_INVALID_OPENING


class Test_outcome_from_signal:
    assert outcome_from_signal.__name__ == "outcome_from_signal"

    @pytest.mark.parametrize("sig,turn", product(PlayerSignal, chess.COLORS))
    def test_mapping_is_complete(self, sig: PlayerSignal, turn: chess.Color):
        outcome_from_signal(sig, turn) # exception will fail the test

    @pytest.mark.parametrize("turn", chess.COLORS)
    def test_failsafe(self, turn: chess.Color):
        with pytest.raises(NotImplementedError):
            outcome_from_signal(cast(Any, -1), turn)
