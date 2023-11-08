"""Tests for san_chess.py"""

import random
from src import san_chess


def test_preset_player():
    """Tests the PresetPlayer class."""

    for _, games in san_chess.conclusive_games.items():
        for movelist in games:
            # Returns preset moves
            p = san_chess.PresetPlayer(movelist)
            for move in movelist:
                sig, suggested_moves = p.suggest_moves()
                assert sig is None
                assert suggested_moves[0] == move
                p.push_moves([move])

            # Sends out of moves signal once the movelist is exceeded.
            sig, moves = p.suggest_moves()
            assert sig == san_chess.PlayerSignal.RESIGNATION
            assert moves == []


def test_random_player():
    """Tests the RandomPlayer class."""

    for seed in range(32):
        rng = random.Random(seed)
        game = san_chess.play_game(san_chess.RandomPlayer(rng=rng, p_invalid=1.0))
        assert game.outcome in san_chess.Outcome.DISQUALIFICATION


def test_play_game():
    """Tests the play_game() function."""

    for outcome, games in san_chess.conclusive_games.items():
        for movelist in games:
            for i in range(min(len(movelist), 16)):
                opening = movelist[:i]  # try prefixes as openings
                game_1p = san_chess.play_game(san_chess.PresetPlayer(movelist), opening_moves=opening)
                game_2p = san_chess.play_game(san_chess.PresetPlayer(movelist), san_chess.PresetPlayer(movelist), opening_moves=opening)
                assert outcome == game_1p.outcome  # outcomes match
                assert outcome == game_2p.outcome  # outcomes match
                assert all(r == 0 for r in game_1p.retries)  # no retries were needed
                assert all(r == 0 for r in game_2p.retries)  # no retries were needed
                assert len(game_1p.retries) == len(game_1p.moves)  # one retry per move
                assert len(game_2p.retries) == len(game_2p.moves)  # one retry per move

        for movelist in games:
            game_1p = san_chess.play_game(san_chess.PresetPlayer([]), opening_moves=movelist)
            game_2p = san_chess.play_game(san_chess.PresetPlayer([]), san_chess.PresetPlayer([]), opening_moves=movelist)
            assert outcome == game_1p.outcome  # outcomes match
            assert outcome == game_2p.outcome  # outcomes match


def test_get_outcome():
    """Tests the get_outcome() function."""

    movelist = "e4 f5 f4 g5 Qh5".split(" ")
    outcome = san_chess.get_outcome(movelist)
    assert outcome == san_chess.Outcome.WHITE_WINS_CHECKMATE
    outcome = san_chess.get_outcome(movelist[:-1])
    assert outcome == san_chess.Outcome.BLACK_WINS_RESIGNATION
    outcome = san_chess.get_outcome(movelist[:-2])
    assert outcome == san_chess.Outcome.WHITE_WINS_RESIGNATION

    movelist = "e9 y5 f4 g5 Qh5".split(" ")
    outcome = san_chess.get_outcome(movelist)
    assert outcome == san_chess.Outcome.ABORT_INVALID_OPENING


def test_outcome():
    """Tests the Outcome enum class."""

    # test from_union_string