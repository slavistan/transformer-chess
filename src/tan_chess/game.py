from __future__ import annotations

from copy import deepcopy
from typing import TypedDict
from enum import Flag, unique, auto

import chess

from .common import (
    TANMove,
    TANMoveList,
    TANPlayer,
    TAN_EOG_CHARS,
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
    DRAW_FIFTY_MOVES = auto()
    DRAW_THREEFOLD_REPETITION = auto()

    ABORT_INVALID_OPENING = auto()
    """Game abort due to providing an invalid sequence of opening moves."""

    # CONTINUEHERE: Allen Code parsen und RESIGNATION Enums hier hinzufügen
    #               - Contract test schreiben der prüft, ob alle Kindsklassen von TANChess Resignation Signale korrekt implementieren
    WHITE_WINS_RESIGNATION = auto()
    BLACK_WINS_RESIGNATION = auto()

    # Aliases to define comfy helpers.
    # TODO: Any von Hand implementieren, mit Test festzurren, dass alle canonicals enthalten sind.
    # ANY = ...

    WHITE_WINS = WHITE_WINS_CHECKMATE | WHITE_WINS_DQ_INVALID_MOVE
    """White won the game."""

    BLACK_WINS = BLACK_WINS_CHECKMATE | BLACK_WINS_DQ_INVALID_MOVE
    """Black won the game."""

    CHECKMATE = WHITE_WINS_CHECKMATE | BLACK_WINS_CHECKMATE
    """Game ended in a checkmate."""

    DRAW_CONCLUSIVE = DRAW_STALEMATE | DRAW_INSUFFICIENT_MATERIAL | DRAW_SEVENTYFIVE_MOVES | DRAW_FIVEFOLD_REPETITION
    """Game ended in a draw, forced by the rules of the game."""

    DRAW_CLAIMED = DRAW_FIFTY_MOVES | DRAW_THREEFOLD_REPETITION
    """Game ended in a draw, claimed by one of the players."""

    DQ_INVALID_MOVE = WHITE_WINS_DQ_INVALID_MOVE | BLACK_WINS_DQ_INVALID_MOVE
    """Game ended due to disqualification, for example due to repeatedly
    suggesting invalid moves."""

    CONCLUSIVE = CHECKMATE | DRAW_CONCLUSIVE

    @staticmethod
    def from_pychess_outcome(outcome: chess.Outcome) -> Outcome:
        """Returns a canonical Outcome from a chess.Outcome."""

        if outcome.winner is not None:
            return Outcome.WHITE_WINS_CHECKMATE if outcome.winner else Outcome.BLACK_WINS_CHECKMATE
        return _termination_to_outcome[outcome.termination]

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
                "outcome": Outcome.from_pychess_outcome(outcome),
            }

        # Get a move suggestion, validate and push it.
        players_idx = board.turn & (len(players) - 1)  # picks the correct player index
        response = players[players_idx].suggest_move()
        if isinstance(response, TANMove):
            move = response
            if not is_valid(move, board):
                outcome = Outcome.BLACK_WINS_DQ_INVALID_MOVE if board.turn else Outcome.WHITE_WINS_DQ_INVALID_MOVE
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
            # Handle signals.
            signal = response  # TODO:
            outcome = Outcome.WHITE_WINS_RESIGNATION if not board.turn else Outcome.BLACK_WINS_RESIGNATION
            return {
                "moves": moves,
                "num_opening_moves": len(opening),
                "outcome": outcome,
            }


def get_outcome(movelist: TANMoveList) -> Outcome:
    """Returns a game's outcome by playing it out by a PresetPlayer.
    Inconclusive games will result in a resignation signal."""
    # TODO: Was, wenn Spiel nicht terminiert? play_game kann hier so nicht verwendet werden.
    # TODO: Was, wenn die Züge ungültig sind?
    game = play_game(PresetPlayer(movelist), opening=movelist)
    return game["outcome"]


def tan_moveline_from_gameline(tan_gameline: str) -> str:
    tan_gameline = tan_gameline.rstrip()
    if tan_gameline.endswith(TAN_EOG_CHARS):
        return tan_gameline[:-2]  # strip eog char and trailing whitespace
    return tan_gameline


# TODO: test
def is_valid(
    move: TANMove | str,
    board: chess.Board,
) -> bool:
    """
    Returns true if the move is valid, given a position. The `board` object is
    not modified.

    :param move: move in TAN format
    :param board: chess.board object representing the position
    """

    # Pychess doesn't offer a method to check the validity of a move in SAN
    # notation, so we have to call push_san() directly and look for exceptions.
    # However, we must not modify the move stack and thus do this on a copy of
    # the board.
    try:
        deepcopy(board).push_san(move)
    except ValueError:
        return False
    return True

    # @staticmethod
    # def from_player_signal(
    #     signal: PlayerSignal,
    #     turn: chess.Color,
    # ) -> Outcome:
    #     """
    #     Converts a player signal into the corresponding game outcome.
    #     """

    #     if signal == PlayerSignal.DIVERGED_FROM_PRESET:
    #         if turn == chess.WHITE:
    #             return Outcome.BLACK_WINS_RESIGNATION_DIVERGED_FROM_PRESET
    #         return Outcome.WHITE_WINS_RESIGNATION_DIVERGED_FROM_PRESET

    #     if signal == PlayerSignal.CONTEXT_OVERFLOW:
    #         if turn == chess.WHITE:
    #             return Outcome.BLACK_WINS_RESIGNATION_CONTEXT_OVERFLOW
    #         return Outcome.WHITE_WINS_RESIGNATION_CONTEXT_OVERFLOW

    #     if signal == PlayerSignal.ABANDONED_GAME:
    #         if turn == chess.WHITE:
    #             return Outcome.BLACK_WINS_RESIGNATION_ABANDONED_GAME
    #         return Outcome.WHITE_WINS_RESIGNATION_ABANDONED_GAME

    #     if signal == PlayerSignal.CANT_CONSTRUCT_MOVE:
    #         if turn == chess.WHITE:
    #             return Outcome.BLACK_WINS_RESIGNATION_CANT_CONSTRUCT_MOVE
    #         return Outcome.WHITE_WINS_RESIGNATION_CANT_CONSTRUCT_MOVE

    #     # TODO: "no legal moves" sollte nicht zum Sieg des anderen führen.
    #     if signal == PlayerSignal.NO_LEGAL_MOVES:
    #         if turn == chess.WHITE:
    #             return Outcome.BLACK_WINS_RESIGNATION_NO_LEGAL_MOVES
    #         return Outcome.WHITE_WINS_RESIGNATION_NO_LEGAL_MOVES

    #     raise NotImplementedError(f"Unknown signal: {signal}")


_termination_to_outcome = {
    chess.Termination.SEVENTYFIVE_MOVES: Outcome.DRAW_SEVENTYFIVE_MOVES,
    chess.Termination.FIVEFOLD_REPETITION: Outcome.DRAW_FIVEFOLD_REPETITION,
    chess.Termination.INSUFFICIENT_MATERIAL: Outcome.DRAW_INSUFFICIENT_MATERIAL,
    chess.Termination.STALEMATE: Outcome.DRAW_STALEMATE,
}
