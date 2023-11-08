"""Basic chess playing abstraction."""
from __future__ import annotations
from collections.abc import Collection
from abc import ABC, abstractmethod
import random
from typing import List, Tuple
import enum
from dataclasses import dataclass
from frozendict import frozendict
import chess
import numpy as np

@enum.unique
class Outcome(enum.Flag):
    """Bitmasks expressing game outcomes."""

    # Canonical values for the exhaustive list of disjunct game results.
    WHITE_WINS_CHECKMATE = enum.auto()
    """White wins by checkmating black."""

    WHITE_WINS_DISQUALIFICATION = enum.auto()
    """White wins due to disqualification of black for suggesting invalid moves."""

    BLACK_WINS_CHECKMATE = enum.auto()
    """Black wins by checkmating white."""

    BLACK_WINS_DISQUALIFICATION = enum.auto()
    """Black wins due to disqualification of white for suggesting invalid moves."""

    DRAW_STALEMATE = enum.auto()
    DRAW_INSUFFICIENT_MATERIAL = enum.auto()
    DRAW_SEVENTYFIVE_MOVES = enum.auto()
    DRAW_FIVEFOLD_REPETITION = enum.auto()
    DRAW_FIFTY_MOVES = enum.auto()
    DRAW_THREEFOLD_REPETITION = enum.auto()

    ABORT_INVALID_OPENING = enum.auto()
    """Game abort due to providing an invalid sequence of opening moves."""

    WHITE_WINS_RESIGNATION = enum.auto()
    BLACK_WINS_RESIGNATION = enum.auto()
    """Player resigns. The specific reason for this depends on the
    implementation of the player itself."""

    # Aliases to define comfy helpers.
    WHITE_WINS = WHITE_WINS_CHECKMATE | WHITE_WINS_DISQUALIFICATION | WHITE_WINS_RESIGNATION
    """White won the game."""

    BLACK_WINS = BLACK_WINS_CHECKMATE | BLACK_WINS_DISQUALIFICATION | BLACK_WINS_RESIGNATION
    """Black won the game."""

    CHECKMATE = WHITE_WINS_CHECKMATE | BLACK_WINS_CHECKMATE
    """Game ended in a checkmate."""

    DRAW_CONCLUSIVE = DRAW_STALEMATE | DRAW_INSUFFICIENT_MATERIAL | DRAW_SEVENTYFIVE_MOVES | DRAW_FIVEFOLD_REPETITION
    """Game ended in a draw, forced by the rules of the game."""

    DRAW_CLAIMED = DRAW_FIFTY_MOVES | DRAW_THREEFOLD_REPETITION
    """Game ended in a draw, claimed by one of the players."""

    DISQUALIFICATION = WHITE_WINS_DISQUALIFICATION | BLACK_WINS_DISQUALIFICATION
    """Game ended due to disqualification, for example due to repeatedly
    suggesting invalid moves."""

    CONCLUSIVE = CHECKMATE | DRAW_CONCLUSIVE

    @staticmethod
    def from_outcome(outcome: chess.Outcome) -> Outcome:
        """Returns a canonical Outcome from a chess.Outcome."""

        if outcome.winner is not None:
            return Outcome.WHITE_WINS_CHECKMATE if outcome.winner else Outcome.BLACK_WINS_CHECKMATE
        return _termination_to_outcome[outcome.termination]

    @staticmethod
    def from_signal(signal: PlayerSignal, turn: chess.Color) -> Outcome:
        """Returns a canonical Outcome from a player signal."""

        if signal == PlayerSignal.RESIGNATION:
            if turn == chess.WHITE:
                return Outcome.BLACK_WINS_RESIGNATION
            return Outcome.WHITE_WINS_RESIGNATION
        raise NotImplementedError

    @staticmethod
    def from_union_string(union_str: str) -> Outcome:
        """Creates an outcome from a |-concatenation of strings."""

        enum_values = union_str.split('|')
        enum_instance = Outcome(0)  # Initialize with no flags
        for enum_name in enum_values:
            try:
                enum_member = Outcome[enum_name.strip()]
                enum_instance |= enum_member
            except KeyError as e:
                raise ValueError(f"Invalid enum value: {enum_name}") from e

        return enum_instance


_termination_to_outcome = {
    chess.Termination.SEVENTYFIVE_MOVES: Outcome.DRAW_SEVENTYFIVE_MOVES,
    chess.Termination.FIVEFOLD_REPETITION: Outcome.DRAW_FIVEFOLD_REPETITION,
    chess.Termination.INSUFFICIENT_MATERIAL: Outcome.DRAW_INSUFFICIENT_MATERIAL,
    chess.Termination.STALEMATE: Outcome.DRAW_STALEMATE,
}


@dataclass
class Game:
    """Outcome, details and statistics of a played game."""

    moves: Collection[str]
    """Moves played in the game."""

    num_opening_moves: int
    """The number of moves that were provided as an opening sequence."""

    retries: Collection[int]
    """Tracks the number of retries that was needed to generate a valid move.
    One entry per move."""

    outcome: Outcome

    @staticmethod
    def summary(games: Collection[Game]):
        """Returns statistics on a collection of games."""

        outcome_counts = {o: 0 for o in Outcome}
        for game in games:
            for outcome in Outcome:
                if game.outcome & outcome:
                    outcome_counts[outcome] += 1
        num_moves = [len(g.moves) for g in games]
        mean, std = np.mean(num_moves), np.std(num_moves)
        return {"outcome_counts": outcome_counts, "num_moves_mean": mean, "num_moves_std": std}


@enum.unique
class PlayerSignal(enum.Enum):
    """Signal communicated by players in addition to their moves."""

    RESIGNATION = enum.auto()


class SANPlayer(ABC):
    """Chess player interface."""

    @abstractmethod
    def __init__(self, movelist: Collection[str] = ()):
        """Initializes player with an optional movelist."""

    @abstractmethod
    def push_moves(self, moves: Collection[str]):
        """Pushes moves to the player's internal move stack."""

    @abstractmethod
    def suggest_moves(self, n: int = 1) -> Tuple[PlayerSignal | None, Collection[str]]:
        """Suggests moves in SAN format. Must not modify move stack."""


class RandomPlayer(SANPlayer):
    """Plays random valid moves, optionally generating invalid moves."""

    def __init__(
        self,
        moves: Collection[str] = (),
        *,
        p_invalid: float = 0.0,
        rng: random.Random = random.Random(),
    ):
        self.rng = rng
        self.p_invalid = p_invalid
        self.board = chess.Board()
        for move in moves:
            self.board.push_san(move)

    def push_moves(self, moves: Collection[str]):
        for move in moves:
            self.board.push_san(move)

    def suggest_moves(self, n: int = 1):
        legal_moves = list(self.board.legal_moves)
        if len(legal_moves) == 0:
            return PlayerSignal.RESIGNATION, ()

        moves: List[str] = []
        for _ in range(n):
            # Generate invalid SAN move.
            if self.rng.random() < self.p_invalid:
                moves.append("Z")
                continue

            move_uci = self.rng.choice(legal_moves)
            # Hack to generate a move in san notation:
            # this will produce a continuation for black: '11...Rg8'
            # or a move for white: '3. Qd3'.
            variation = self.board.variation_san([move_uci])
            movepos = variation.rfind(".")
            if variation[movepos + 1] == " ":
                move = variation[movepos + 2 :]
            else:
                move = variation[movepos + 1 :]
            moves.append(move)
        return None, moves


class PresetPlayer(SANPlayer):
    """Plays preset moves. Used for testing via fixed sequences of moves.
    Throws if game exceeds preset moves."""

    def __init__(
        self,
        moves: Collection[str] = (),
    ):
        self.moves = list(moves)
        self.move_idx = 0  # current move

    def push_moves(self, moves: Collection[str]):
        self.move_idx += len(moves)

    def suggest_moves(self, n: int = 1):
        if self.move_idx >= len(self.moves):
            moves: List[str] = []
            return PlayerSignal.RESIGNATION, moves
        return None, [self.moves[self.move_idx]] * n


def play_game(
    white: SANPlayer,
    black: SANPlayer | None = None,
    *,
    opening_moves: Collection[str] = (),
    num_retries: int = 0,
) -> Game:
    """Plays a game with one or two players, returning a result and game
    statistics. No draws can be claimed to make the outcome deterministic."""

    moves = list(opening_moves)
    retries = [0] * len(opening_moves)
    players = ([black] if black is not None else []) + [white]  # used to index via booleans, as used by python's chess library.

    # Set up board and play opening moves.
    board = chess.Board()
    try:
        for move in opening_moves:
            board.push_san(move)
    except ValueError:
        return Game(moves, len(opening_moves), retries, Outcome.ABORT_INVALID_OPENING)
    for p in players:
        p.push_moves(moves)

    while True:
        # Check if game has ended.
        outcome = board.outcome()
        if outcome is not None:
            return Game(moves, len(opening_moves), retries, Outcome.from_outcome(outcome))

        # Get a move suggestions, validate and push it.
        found_move_after_retries = False
        players_idx = board.turn & (len(players) - 1)  # picks the correct player index
        signal, move_suggestions = players[players_idx].suggest_moves(num_retries + 1)
        if signal is not None:
            return Game(moves, len(opening_moves), retries, Outcome.from_signal(signal, board.turn))
        retry = 0
        for retry, move in enumerate(move_suggestions):
            try:
                board.push_san(move)
            except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
                continue

            moves.append(move)
            for p in players:
                p.push_moves([move])
            found_move_after_retries = True
            break

        if not found_move_after_retries:
            outcome = Outcome.BLACK_WINS_DISQUALIFICATION if board.turn else Outcome.WHITE_WINS_DISQUALIFICATION
            return Game(moves, len(opening_moves), retries, outcome)

        retries.append(retry)


def get_outcome(san_movelist: Collection[str]) -> Outcome:
    """Returns a game's outcome by playing it out by a PresetPlayer.
    Inconclusive games will result in a resignation signal."""
    game = play_game(PresetPlayer([]), opening_moves=san_movelist)
    return game.outcome


conclusive_games = frozendict(
    {
        Outcome.WHITE_WINS_CHECKMATE: [
            # Fool's mate
            "e4 f5 f4 g5 Qh5".split(" "),
            # Scholar's mate
            "e4 e5 Bc4 Nc6 Qh5 Nf6 Qxf7".split(" "),
        ],
        Outcome.BLACK_WINS_CHECKMATE: [
            # Fool's mate
            "g4 e5 f4 Qh4".split(" "),
            # Scholar's mate
            "e4 e5 Nc3 Bc5 Bb5 Qh4 Nf3 Qxf2".split(" "),
        ],
        Outcome.DRAW_STALEMATE: [
            # https://lichess.org/VO3jq34v#68
            "d4 d5 Nc3 Nf6 e3 Bg4 Nf3 Nc6 Qd3 e6 Ne5 Qe7 e4 Qb4 h3 Bh5 exd5 Nxd5 Qb5 Qxb5 Nxc6 Qxc6 Kd2 Bb4 f3 O-O g4 Bg6 f4 Rfe8 a3 Ba5 h4 Nxf4 d5 exd5 Ra2 d4 Kd1 dxc3 Be3 cxb2 Bd3 Qb6 Bxg6 Nxg6 a4 b1=Q Ke2 Qxh1 g5 Qhb1 c3 Qxa2 Ke1 Qxa4 Kf2 Qab5 Bc5 Q6xc5 Kf3 Re3 Kg4 Rae8 Kh5 R8e4 c4 Qbxc4".split(" ")
        ],
        Outcome.DRAW_INSUFFICIENT_MATERIAL: [
            "e4 d5 exd5 Qxd5 Nc3 Qxg2 Qh5 Qxh1 Qxh7 Qxh2 Qxh8 Qxg1 Qxg8 Nc6 d3 Qxf1 Kd2 Qxc1 Ke2 Qxa1 Kd2 Qxa2 Kc1 Qxb2 Kd1 Qxc3 Ke2 Qxc2 Ke1 Qxd3 f3 Qxf3 Qxg7 Qg2 Qxf8 Kd7 Qxf7 Qg3 Kd2 b6 Qf8 Ke6 Qxc8 Kf6 Qxa8 Kg7 Qxa7 Kf8 Qxb6 Kg7 Qxc7 Kg8 Qxc6 Kg7 Qd6 Kg8 Qxe7 Qg7 Qxg7 Kxg7".split(" ")
        ],
        Outcome.DRAW_FIVEFOLD_REPETITION: [
            # https://lichess.org/UzKC66ai
            "e4 d6 d4 Nf6 Nc3 g5 Bxg5 h6 Bxf6 exf6 Nf3 Nc6 Bb5 Bd7 Qd2 Qe7 O-O-O O-O-O Rhe1 a6 Ba4 b5 Bb3 Na5 Nd5 Nxb3 cxb3 Qe6 Qc3 c6 Kb1 Kb7 Rc1 Rc8 Qa5 cxd5 exd5 Qxd5 Rxc8 Bxc8 Rc1 Bg7 Qc7 Ka8 Qa5 Bf5 Ka1 Rc8 Qxa6 Kb8 Qb6 Ka8 Qa6 Kb8 Qb6 Ka8 Qa6 Kb8 Qb6 Ka8 Qa6 Kb8 Qb6 Ka8 Qa6".split(" ")
        ],
        Outcome.DRAW_SEVENTYFIVE_MOVES: [
            "Nc3 c5 b4 cxb4 e4 Na6 Bd3 f6 f4 b6 Bb2 Bb7 Ba3 Rb8 Bxb4 e6 Bxa6 d6 Ba3 Qc7 Bxb7 Qd8 Ba6 g5 Bb5+ Qd7 g4 h6 Nf3 Kf7 fxg5 Rd8 gxh6 Ke8 Ng5 Bxh6 Nh7 Qxb5 Qe2 Bxd2+ Kxd2 Kd7 Bc1 Kc8 g5 Ne7 Qd1 a6 Na4 Qd3+ Kxd3 Nd5 Bb2 Rde8 Rc1 Kd8 Rg1 Kc7 Qg4 Ne7 Qd1 Kd7 Nc3 Ra8 Qd2 Rxh7 Rg3 Rc8 Nb1 Ng8 Rh1 Ke7 c4 Rf8 h4 a5 Qd1 fxg5 h5 Rhf7 Rgg1 Rh7 Be5 Nh6 Kc2 Rf6 Nc3 Ke8 Na4 Rhf7 Rxg5 Rg7 Rf5 Kd7 Rg5 Nf5 Nb2 Kd8 Qd2 Kd7 Rh3 Nd4+ Kd3 Nc6 Qc2 Rf1 Re3 Rh1 Qf2 a4 Bd4 Nxd4 Rf3 Nf5 Qh2 Rg1 Qh3 Ra1 Rfg3 Rb1 Rg1 Nd4 Qe3 d5 a3 Nc2 Qxb6 Ne3 h6 Re7 Kxe3 Kc8 Rxd5 Rc7 Qxe6+ Kb7 Qc8+ Rxc8 Rg2 Rb8 Rd6 Rg8 Kd4 Rg7 Kd5 Kb8 Rg4 Rg5+ Kc6 Rg6 Re6 Re1 Rf4 Rf6 Rfxf6 Rh1 Rf7 Rh5 Ref6 Rd5 Nd3 Ra5 e5 Rc5+ Kd6 Rxe5 Rh7 Re7 Rf1 Re8 Re7 Rg8 Nb4 Rg7 Rf5 Rg1 Rb7+ Ka8 Nc2 Ra1 Re7 Rh1 Ne3 Kb8 Nc2 Ka8 Na1 Rh2 Kd7 Kb7 Rf8 Rh4 Rfe8 Re4 Kd6+ Kb6 Rg7 Rf4 Re1 Ka6 Rd1 Rf5 Rd4 Rd5+ cxd5 Ka5 Rf4 Ka6 Rg8 Ka7 Rc4 Kb7 Rh4 Ka6 Rhg4 Kb5 Rd4 Ka5 Re8 Ka6 Rg4 Ka5 Rf8 Ka6 Rf3 Kb5 Ke7 Ka6 Rb3 Ka5 Ke8 Ka6 Nc2 Ka7 Rh4 Ka6 Kf8 axb3 Re4 bxc2 Ke8 Ka5 Rg4 c1=R Rg6 Rh1 Kd7 Rd1 Kc8 Rb1 a4 Rb6 h7 Rd6 Rf6 Kxa4 Rf2 Kb5 Rd2 Kc4 Re2 Rf6 Re7 Rd6 Rd7 Kd3 Ra7 Rf6 Rb7 Rf7 Rb4 Rxh7 Kb8 Rh5 Rc4 Kd2 Rf4 Rh3 Ra4 Ke3 Re4+ Kd3 Re6 Rg3 Re7 Rh3 d6 Rh7 Rb7 Re7 Rb6 Kd2 Rc6 Rg7 Rc4 Kd3 Rc6 Ke4 Rc3 Rg3 Rxg3 Kd4 Rg4+ Kc5 Rg2 Kb5 Rg5+ Kb4 Rh5 Ka3 Ka7 Kb3 Re5 Ka4 Rc5 Ka3 Rh5 Kb2 Ka8 Kb3 Rc5 Ka2 Rc4 Kb1 Rh4 Ka2 Rh5 Kb3 Kb7 Ka3 Kb8 Ka2 Rh6 Kb3 Rh5 Kc3 Rh8 Kc2 Rh1 Kb3 d7 Kc3 d8=N Kb3 Rg1 Ka4 Rh1 Kb3 Kc8 Ka4 Kb8 Kb3 Re1 Ka4 Ne6 Kb4 Nf4 Ka5 Ng2 Kb6 Re3 Kb5 Kb7 Kc5 Rc3+ Kb5 Ra3 Kc5 Rb3 Kd6 Rb4 Ke7 Kb6 Kf6 Rb1 Ke6 Kb7 Kf5 Re1 Kf6 Kc6 Kg6 Re5 Kf6 Re4 Kg7 Re5 Kh6 Re3 Kg6 Rd3 Kf6 Ra3 Kg7 Nf4 Kh8 Ng2 Kg8 Ne3 Kf8 Kc7 Kf7 Kc8 Kg8 Kc7 Kf8 Ra8+ Kg7 Rb8 Kh7 Kc8 Kg7 Rb4 Kf7 Rb6 Ke7 Kb7 Ke8 Ka7 Ke7 Rf6 Kd7 Rg6 Ke7 Rg1 Ke8 Kb7 Ke7 Rg6 Kd7 Ka8 Kd8 Rb6 Ke8 Rb8+ Kd7 Rh8 Kc7 Rh2 Kc6 Nf1 Kc5 Ng3 Kc4 Nh1 Kb4 Kb7 Ka4 Kb8 Ka5 Ka7 Ka4 Rd2 Kb3 Ka6 Kb4 Rb2+ Kc5 Rf2 Kc4 Rc2+ Kb3 Rc4 Kb2 Ra4 Kb3 Ra5 Kb2 Nf2 Kc3 Rd5 Kb3 Rd2 Kc4 Rd8 Kc3 Re8 Kb2 Re1 Kc3 Rg1 Kb2 Rd1 Kb3 Rd2 Ka4 Rd5 Kb3 Ka5 Kc3 Re5 Kb2 Rg5".split(" ")
        ],
    }
)
