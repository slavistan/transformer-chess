from __future__ import annotations

from abc import ABC, abstractmethod
import os
from copy import deepcopy
from enum import Enum
from typing import Sequence, List
import multiprocessing as mp
import subprocess

import requests
import chess

TANMove = str
TANMoveList = Sequence[TANMove]
TANMoveLine = str
TANGameLine = str

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
)

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

    class ResignationReason(Enum):
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


def get_legal_moves(
    board: chess.Board,
) -> List[TANMove]:
    """
    Give a position represented by `board`, returns a list of valid moves in
    TAN format.
    """

    legal_moves = [uci_to_tan(uci_move, board) for uci_move in board.legal_moves]
    return legal_moves


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

    # Remove any SAN annotations.
    variation = variation.rstrip("?!+#")

    movepos = variation.rfind(".")
    if variation[movepos + 1] == " ":
        move = variation[movepos + 2 :]
    else:
        move = variation[movepos + 1 :]

    return move


def trim_san_move(
    san_move: str,
) -> TANMove:
    """
    Removes annotations from a move in SAN format, returning a move in TAN
    format.
    """

    return san_move.rstrip("?!+#")


def tan_moveline_from_gameline(
    tan_gameline: TANGameLine,
) -> TANMoveLine:
    """
    Given a valid gameline in TAN format returns the corresponding moveline in
    TAN format.

    Movelines are returned unmodified.
    """

    tan_gameline = tan_gameline.rstrip()
    if tan_gameline.endswith(TAN_EOG_CHARS):
        return tan_gameline[:-2]  # strip eog char and trailing whitespace
    return tan_gameline


def is_valid_move(
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
    # However, we must not modify the move stack and thus we do this on a copy
    # of the board.
    try:
        deepcopy(board).push_san(move)
    except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
        return False
    return True


def is_valid_moveline(
    moveline: TANMoveLine,
) -> bool:
    """
    Returns true if the moveline is valid, as determined by the following
    defining criteria:
      - moveline is a string
      - moveline must consist of legal characters only (e.g. no SAN annotations)
      - moves must be separated by a single whitespace
      - moveline must not have surrounding whitespaces
      - moveline must constitute a series of valid moves
    """

    # Moveline may be empty.
    if len(moveline) == 0:
        return True

    # Moveline must be a string.
    if not isinstance(moveline, TANMoveLine):
        return False

    # Moveline must only consists of allowed characters.
    if set(moveline) | set(TAN_MOVELINE_CHARS) != set(TAN_MOVELINE_CHARS):
        return False

    # Moveline must separate moves by exactly one whitespace.
    if moveline.find("  ") != -1:
        return False

    # Moveline must not be surrounded by spurious whitespaces.
    if moveline.strip() != moveline:
        return False

    # Moveline must constitute a series of valid moves.
    board = chess.Board()
    try:
        for m in moveline.split(" "):
            board.push_san(m)
    except (chess.AmbiguousMoveError, chess.InvalidMoveError, chess.IllegalMoveError):
        return False

    return True


def is_valid_movelist(
    movelist: TANMoveList,
) -> bool:
    return is_valid_moveline(" ".join(movelist))


def is_conclusive_movelist(
    movelist: TANMoveList,
) -> bool:
    """
    Returns true iff the last move in `movelist` concludes the game.
    """

    board = chess.Board()
    for m in movelist:
        board.push_san(m)
    is_eog = board.outcome() is not None
    return is_eog


# TODO: test
def is_game_ending_move(
    move: TANMove,
    board: chess.Board,
) -> bool:
    """
    Returns true iff the move would end the game. The move must be valid.

    The chess.Board object is not modified.
    """

    board = deepcopy(board)
    board.push_san(move)
    return board.outcome() is not None


def movelist_to_moveline(
    movelist: TANMoveList,
) -> TANMoveLine:
    moveline = " ".join(movelist)
    return moveline


def moveline_to_movelist(
    moveline: TANMoveLine,
) -> TANMoveList:
    movelist = moveline.split(" ")
    return movelist


def view_game(
    moveline: TANMoveLine | TANMoveList,
) -> str:
    """
    Views a game in the browser.

    The game is uploaded to the lichess servers via their public API and the
    browser is opened afterwards. Requires the $BROWSER envvar to be set.

    Returns the url of the game on the lichess website.
    """

    if not isinstance(moveline, TANMoveLine):
        moveline = " ".join(moveline)

    post_game_url = "https://lichess.org/api/import"
    headers = {"accept": "application/json"}
    data = {"pgn": moveline}
    response = requests.post(
        url=post_game_url,
        headers=headers,
        data=data,
        timeout=5,
    )
    game_url = response.json()["url"]

    if os.environ.get("BROWSER"):
        p = mp.Process(
            target=_view_game_open_browser,
            args=(os.environ["BROWSER"], game_url),
            daemon=True,
        )
        p.start()

    return game_url


def _view_game_open_browser(
    browser_name: str,
    url: str,
):
    subprocess.run([browser_name, url], check=False)
