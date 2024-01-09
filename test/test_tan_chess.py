"""Tests for tan_chess.py"""

from itertools import product
from typing import cast, Any

import pytest
import chess

from src.tan_chess import PlayerSignal, play_game, get_outcome, conclusive_games, Outcome
from src.tan_chess.preset_player import PresetPlayer
from src.tan_chess.random_player import RandomPlayer


class Test_RandomPlayer:
    assert RandomPlayer.__name__ == "RandomPlayer"

    # TODO: Test out of moves signal
    # TODO: Test basic functionality (one move positions)

class Test_play_game:
    assert play_game.__name__ == "play_game"

    def test_default(self):
        """Tests the play_game() function."""

        for outcome, games in conclusive_games.items():
            for movelist in games:
                for i in range(min(len(movelist), 16)):
                    opening = movelist[:i]  # try prefixes as openings
                    game_1p = play_game(PresetPlayer(movelist), opening=opening)
                    game_2p = play_game(PresetPlayer(movelist), PresetPlayer(movelist), opening=opening)
                    assert outcome == game_1p["outcome"]  # outcomes match
                    assert outcome == game_2p["outcome"]  # outcomes match

            for movelist in games:
                game_1p = play_game(PresetPlayer([]), opening=movelist)
                game_2p = play_game(PresetPlayer([]), PresetPlayer([]), opening=movelist)
                assert outcome == game_1p["outcome"]  # outcomes match
                assert outcome == game_2p["outcome"]  # outcomes match

def test_get_outcome():

    fools_mate = conclusive_games[Outcome.WHITE_WINS_CHECKMATE][0]
    outcome = get_outcome(fools_mate)
    assert outcome == Outcome.WHITE_WINS_CHECKMATE
    outcome = get_outcome(fools_mate[:-1])
    assert outcome == Outcome.BLACK_WINS_RESIGNATION_DIVERGED_FROM_PRESET
    outcome = get_outcome(fools_mate[:-2])
    assert outcome == Outcome.WHITE_WINS_RESIGNATION_DIVERGED_FROM_PRESET

    invalid_moves = "e9 y5 f4 g5 Qh5".split(" ")
    outcome = get_outcome(invalid_moves)
    assert outcome == Outcome.ABORT_INVALID_OPENING


# TODO: verwende Outcome.from_signal()
class Test_outcome_from_signal:
    assert outcome_from_signal.__name__ == "outcome_from_signal"

    @pytest.mark.parametrize("sig,turn", product(PlayerSignal, chess.COLORS))
    def test_mapping_is_complete(self, sig: PlayerSignal, turn: chess.Color):
        outcome_from_signal(sig, turn) # exception will fail the test

    @pytest.mark.parametrize("turn", chess.COLORS)
    def test_failsafe(self, turn: chess.Color):
        with pytest.raises(NotImplementedError):
            outcome_from_signal(cast(Any, -1), turn)
