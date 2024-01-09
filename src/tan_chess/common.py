from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Sequence

import chess

TANMove = str
TANMoveList = Sequence[TANMove]

# Characters required to express a single move in TAN format, e.g. 'a4', 'Qxb4'
# or 'O-O-O'.
# fmt: off
TAN_MOVE_CHARS = (
    "1", "2", "3", "4", "5", "6", "7", "8",  # ranks
    "a", "b", "c", "d", "e", "f", "g", "h",  # files
    "B", "K", "N", "Q", "R",                 # pieces
    "x", "=", "O", "-",                      # captures, promotion, castling
)
# fmt: on

# A moveline is the whitespace-separated concatenation of a game's movelist,
# not including the game's result, e.g.
#
#   e4 f6 d4 g5 Qh5
#
TAN_MOVELINE_CHARS = (" ",) + TAN_MOVE_CHARS

# End of game identifiers. Single character abbreviations of '1-0', '0-1' and
# '1/2-1/2'.
TAN_EOG_CHARS = (
    "W",  # white wins
    "S",  # black wins
    "U",  # draw
)  # white wins  # black wins  # draw

# A gameline is the whitespace-separated concatenation of a game's movelist,
# including the end of game identifier, e.g.
#
#   e4 f6 d4 g5 Qh5 W
#
TAN_GAMELINE_CHARS = TAN_EOG_CHARS + TAN_MOVELINE_CHARS

# The maximum length of a single move in TAN format, e.g. 'Qa1xg3'. Note that
# the lichess databases don't denote en-passent at all, thus we don't include
# it in the TAN format.
TAN_MAX_MOVE_LEN = len("Qa1xg3")


class TANPlayer(ABC):
    """Chess player interface."""

    # TODO: info() Methode:
    #       - name (testen, dass Namen eindeutig sind)

    class ResignationReason(StrEnum):
        """
        Resignation signals of player. Subclasses must override this if they
        can resign.
        """

    @abstractmethod
    def push_moves(self, movelist: TANMoveList) -> TANPlayer:
        """
        Pushes moves to the player's internal move stack. The moves are
        guaranteed to be valid.

        Returns self.
        """

    @abstractmethod
    def suggest_move(self) -> TANPlayer.ResignationReason | TANMove:
        """
        Suggests a move in TAN format or returns a signal. Must not modify move stack.
        """

    @abstractmethod
    def reset(self) -> TANPlayer:
        """
        Resets the player's internal state.

        Returns self.
        """


def uci_to_tan(
    move_uci: chess.Move,
    board: chess.Board,
) -> TANMove:
    """
    Given a move in UCI format and a board, generates the equivalent move in
    TAN format.
    """

    # Hack to generate a move in san notation: this will produce a continuation
    # for black, e.g. '11...Rg8' or a move for white: '3. Qd3'.
    variation = board.variation_san([move_uci])
    movepos = variation.rfind(".")
    if variation[movepos + 1] == " ":
        move = variation[movepos + 2 :]
    else:
        move = variation[movepos + 1 :]

    return move
