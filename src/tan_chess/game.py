from __future__ import annotations

from copy import deepcopy
from typing import TypedDict

from enum import Flag, unique, auto

import chess

from .common import (
    TANMove,
    TANMoveList,
    TANPlayer,
    is_valid_move,
)
from .players import PresetPlayer


@unique
class Outcome(Flag):
    """Bitmasks expressing game outcomes."""

    # Canonical values for the exhaustive list of disjunct game results.
    WHITE_WINS_CHECKMATE = auto()
    """White wins by checkmating black."""

    BLACK_WINS_CHECKMATE = auto()
    """Black wins by checkmating white."""

    WHITE_WINS_DQ_INVALID_MOVE = auto()
    """White wins due to disqualification of black for suggesting an invalid move."""

    BLACK_WINS_DQ_INVALID_MOVE = auto()
    """Black wins due to disqualification of white for suggesting an invalid move."""

    DRAW_STALEMATE = auto()
    DRAW_INSUFFICIENT_MATERIAL = auto()
    DRAW_SEVENTYFIVE_MOVES = auto()
    DRAW_FIVEFOLD_REPETITION = auto()

    ABORT_INVALID_OPENING = auto()
    """Game abort due to providing an invalid sequence of opening moves."""

    INCOMPLETE_GAME = auto()
    """Game didn't end conclusively"""

    # The following entries represent outcomes produced by the player specific
    # resignation reasons. Correspondence between the player classes'
    # resignation reason enums and the entries below is asserted by tests.
    WHITE_WINS_RESIGNATION_PRESETPLAYER_DIVERGED_FROM_PRESET = auto()
    WHITE_WINS_RESIGNATION_PRESETPLAYER_EXCEEDED_PRESET = auto()
    WHITE_WINS_RESIGNATION_RANDOMPLAYER_NO_LEGAL_MOVES = auto()
    WHITE_WINS_RESIGNATION_GUIPLAYER_ABANDONED_GAME = auto()
    WHITE_WINS_RESIGNATION_TRANSFORMERPLAYER_CONTEXT_OVERFLOW = auto()
    WHITE_WINS_RESIGNATION_TRANSFORMERPLAYER_CANT_CONSTRUCT_VALID_MOVE = auto()
    BLACK_WINS_RESIGNATION_PRESETPLAYER_DIVERGED_FROM_PRESET = auto()
    BLACK_WINS_RESIGNATION_PRESETPLAYER_EXCEEDED_PRESET = auto()
    BLACK_WINS_RESIGNATION_RANDOMPLAYER_NO_LEGAL_MOVES = auto()
    BLACK_WINS_RESIGNATION_GUIPLAYER_ABANDONED_GAME = auto()
    BLACK_WINS_RESIGNATION_TRANSFORMERPLAYER_CONTEXT_OVERFLOW = auto()
    BLACK_WINS_RESIGNATION_TRANSFORMERPLAYER_CANT_CONSTRUCT_VALID_MOVE = auto()

    # Aliases to define comfy helpers.
    # fmt: off
    WHITE_WINS = WHITE_WINS_CHECKMATE | \
        WHITE_WINS_DQ_INVALID_MOVE | \
        WHITE_WINS_RESIGNATION_PRESETPLAYER_DIVERGED_FROM_PRESET  | \
        WHITE_WINS_RESIGNATION_RANDOMPLAYER_NO_LEGAL_MOVES  | \
        WHITE_WINS_RESIGNATION_GUIPLAYER_ABANDONED_GAME  | \
        WHITE_WINS_RESIGNATION_TRANSFORMERPLAYER_CONTEXT_OVERFLOW  | \
        WHITE_WINS_RESIGNATION_TRANSFORMERPLAYER_CANT_CONSTRUCT_VALID_MOVE
    """White won the game."""

    BLACK_WINS = BLACK_WINS_CHECKMATE | \
        BLACK_WINS_DQ_INVALID_MOVE | \
        BLACK_WINS_RESIGNATION_PRESETPLAYER_DIVERGED_FROM_PRESET  | \
        BLACK_WINS_RESIGNATION_RANDOMPLAYER_NO_LEGAL_MOVES  | \
        BLACK_WINS_RESIGNATION_GUIPLAYER_ABANDONED_GAME  | \
        BLACK_WINS_RESIGNATION_TRANSFORMERPLAYER_CONTEXT_OVERFLOW  | \
        BLACK_WINS_RESIGNATION_TRANSFORMERPLAYER_CANT_CONSTRUCT_VALID_MOVE
    """Black won the game."""
    # fmt: on

    CHECKMATE = WHITE_WINS_CHECKMATE | BLACK_WINS_CHECKMATE
    """Game ended in a checkmate."""

    DRAW_CONCLUSIVE = DRAW_STALEMATE | DRAW_INSUFFICIENT_MATERIAL | DRAW_SEVENTYFIVE_MOVES | DRAW_FIVEFOLD_REPETITION
    """Game ended in a draw, forced by the rules of the game."""

    DQ_INVALID_MOVE = WHITE_WINS_DQ_INVALID_MOVE | BLACK_WINS_DQ_INVALID_MOVE
    """Game ended due to disqualification, for example due to repeatedly
    suggesting invalid moves."""

    CONCLUSIVE = CHECKMATE | DRAW_CONCLUSIVE

    @staticmethod
    def from_python_chess_outcome(outcome: chess.Outcome) -> Outcome:
        """Returns a canonical Outcome from a chess.Outcome."""

        python_chess_draws_to_outcome = {
            chess.Termination.SEVENTYFIVE_MOVES: Outcome.DRAW_SEVENTYFIVE_MOVES,
            chess.Termination.FIVEFOLD_REPETITION: Outcome.DRAW_FIVEFOLD_REPETITION,
            chess.Termination.INSUFFICIENT_MATERIAL: Outcome.DRAW_INSUFFICIENT_MATERIAL,
            chess.Termination.STALEMATE: Outcome.DRAW_STALEMATE,
        }
        if outcome.winner is not None:
            return Outcome.WHITE_WINS_CHECKMATE if outcome.winner else Outcome.BLACK_WINS_CHECKMATE
        return python_chess_draws_to_outcome[outcome.termination]

    @staticmethod
    def from_union_string(union_str: str) -> Outcome:
        """Creates an outcome from a |-concatenation of strings."""

        enum_values = union_str.split("|")
        enum_instance = Outcome(0)  # Initialize with no flags
        for enum_name in enum_values:
            try:
                enum_member = Outcome[enum_name.strip()]
                enum_instance |= enum_member
            except KeyError as e:
                raise ValueError(f"Invalid enum value: {enum_name}") from e

        return enum_instance


class Game(TypedDict):
    """Outcome, details and statistics of a played game."""

    moves: TANMoveList
    """Moves played in the game."""

    num_opening_moves: int
    """The number of moves that were provided as an opening sequence."""

    outcome: Outcome


def play_game(
    white: TANPlayer,
    black: TANPlayer | None = None,
    *,
    opening: TANMoveList = (),
) -> Game:
    """Plays a game with one or two players, returning a result and game
    statistics. No draws can be claimed to make the outcome deterministic."""

    # TODO: Game sollte einfach das chess.Board zurückgeben, aus dem die Züge rekonstruiert werden können.
    #       - Teste ob das geht, und Helper schreiben, der aus board die SAN Züge extrahiert
    #       - Muss gucken, wie json-serialisierbarkeit erhalten werden kann

    # TODO: Game muss jetzt optionale Spielerstatistiken speichern können

    # Players used to index via booleans, as used by python's chess library.
    # (black, white) for two players, otherwise (white)
    players = ([black] if black is not None else []) + [white]

    # Set up board and play opening moves.
    board = chess.Board()
    try:
        for move in opening:
            board.push_san(move)
    except ValueError:
        return {
            "moves": opening,
            "num_opening_moves": len(opening),
            "outcome": Outcome.ABORT_INVALID_OPENING,
        }

    for p in players:
        p.push_moves(opening)

    moves = deepcopy(list(opening))
    while True:
        # Check if game has ended.
        outcome = board.outcome()
        if outcome is not None:
            return {
                "moves": moves,
                "num_opening_moves": len(opening),
                "outcome": Outcome.from_python_chess_outcome(outcome),
            }

        players_idx = board.turn & (len(players) - 1)  # picks the correct player index
        response = players[players_idx].suggest_move()
        if isinstance(response, TANMove):
            # Player returned a move.
            move = response
            if not is_valid_move(move, board):
                outcome_name = f"{'WHITE' if not board.turn else 'BLACK'}_WINS_DQ_INVALID_MOVE"
                outcome = Outcome[outcome_name]
                return {
                    "moves": moves,
                    "num_opening_moves": len(opening),
                    "outcome": outcome,
                }

            moves.append(move)
            for p in players:
                p.push_moves([move])
            board.push_san(move)
        else:
            # Player resigned.
            resignation_reason = response
            outcome_name = f"{'WHITE' if not board.turn else 'BLACK'}_WINS_RESIGNATION_{players[players_idx].__class__.__name__.upper()}_{resignation_reason.name}"
            outcome = Outcome[outcome_name]
            return {
                "moves": moves,
                "num_opening_moves": len(opening),
                "outcome": outcome,
            }


def get_outcome(
    movelist: TANMoveList,
) -> Outcome:
    """
    Returns a game's outcome by playing it out by a PresetPlayer.
    Inconclusive games will result in a resignation signal.
    """

    # We retrieve the outcome of a game by playing it out using a PresetPlayer.
    # Incomplete games are detected by the player resigning due to exceeding
    # the preset list of moves. Invalid games are detected by play_game itself.
    game = play_game(PresetPlayer(movelist), opening=movelist)
    if game["outcome"] in (Outcome.WHITE_WINS_RESIGNATION_PRESETPLAYER_EXCEEDED_PRESET, Outcome.BLACK_WINS_RESIGNATION_PRESETPLAYER_EXCEEDED_PRESET):
        return Outcome.INCOMPLETE_GAME
    return game["outcome"]
