"""Basic chess playing abstraction."""
from __future__ import annotations
import tkinter as tk
from copy import deepcopy
from collections.abc import Collection
import random
from abc import abstractmethod
from typing import Sequence, Literal, Protocol, TypedDict, NewType
import io
import enum
from frozendict import frozendict
import chess
import chess.svg
import cairosvg
from PIL import Image, ImageTk

# Characters required to express a single move in TAN format, e.g. 'a4', 'Qxb4'
# or 'O-O-O'.
TAN_MOVE_CHARS = ("1", "2", "3", "4", "5", "6", "7", "8", "a", "b", "c", "d", "e", "f", "g", "h", "B", "K", "N", "Q", "R", "x", "=", "O", "-")  # Ranks  # Files  # Pieces  # Captures, promotion, castling

# A moveline is the whitespace-separated concatenation of a game's movelist,
# not including the game's result, e.g.
#
#   e4 f6 d4 g5 Qh5
#
TAN_MOVELINE_CHARS = (" ",) + TAN_MOVE_CHARS

# End of game identifiers. Single character abbreviations of '1-0', '0-1' and
# '1/2-1/2'.
TAN_EOG_CHARS = ("W", "S", "U")  # white wins  # black wins  # draw

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

SAN_MOVE_ANNOTATION_POSTFIX = "?!+#"

# Declarative types representing a move in standard or trimmed algebraic
# notation
SANMove = NewType("SANMove", str)
TANMove = NewType("TANMove", SANMove)

# Movelists represent a sequence of moves from the starting position.
SANMoveList = NewType("SANMoveList", Sequence[SANMove])
TANMoveList = NewType("TANMoveList", Sequence[TANMove])


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

    BLACK_WINS_DQ_INVALID_MOVE = enum.auto()
    """Black wins due to disqualification of white for suggesting invalid moves."""

    DRAW_STALEMATE = enum.auto()
    DRAW_INSUFFICIENT_MATERIAL = enum.auto()
    DRAW_SEVENTYFIVE_MOVES = enum.auto()
    DRAW_FIVEFOLD_REPETITION = enum.auto()
    DRAW_FIFTY_MOVES = enum.auto()
    DRAW_THREEFOLD_REPETITION = enum.auto()

    ABORT_INVALID_OPENING = enum.auto()
    """Game abort due to providing an invalid sequence of opening moves."""

    WHITE_WINS_RESIGNATION_DIVERGED_FROM_PRESET = enum.auto()
    BLACK_WINS_RESIGNATION_DIVERGED_FROM_PRESET = enum.auto()
    """See corresponding PlayerSignal."""

    WHITE_WINS_RESIGNATION_CONTEXT_OVERFLOW = enum.auto()
    BLACK_WINS_RESIGNATION_CONTEXT_OVERFLOW = enum.auto()
    """See corresponding PlayerSignal."""

    WHITE_WINS_RESIGNATION_CANT_CONSTRUCT_MOVE = enum.auto()
    BLACK_WINS_RESIGNATION_CANT_CONSTRUCT_MOVE = enum.auto()
    """See corresponding PlayerSignal."""

    WHITE_WINS_RESIGNATION_ABANDONED_GAME = enum.auto()
    BLACK_WINS_RESIGNATION_ABANDONED_GAME = enum.auto()
    """See corresponding PlayerSignal."""

    WHITE_WINS_RESIGNATION_NO_LEGAL_MOVES = enum.auto()
    BLACK_WINS_RESIGNATION_NO_LEGAL_MOVES = enum.auto()
    """See corresponding PlayerSignal."""

    # Aliases to define comfy helpers.
    # TODO: Any von Hand implementieren, mit Test festzurren, dass alle canonicals enthalten sind.
    # ANY = ...

    WHITE_WINS = WHITE_WINS_CHECKMATE | WHITE_WINS_DISQUALIFICATION | WHITE_WINS_RESIGNATION_DIVERGED_FROM_PRESET | WHITE_WINS_RESIGNATION_CONTEXT_OVERFLOW | WHITE_WINS_RESIGNATION_ABANDONED_GAME
    """White won the game."""

    BLACK_WINS = BLACK_WINS_CHECKMATE | BLACK_WINS_DQ_INVALID_MOVE | BLACK_WINS_RESIGNATION_DIVERGED_FROM_PRESET | BLACK_WINS_RESIGNATION_CONTEXT_OVERFLOW | BLACK_WINS_RESIGNATION_ABANDONED_GAME
    """Black won the game."""

    CHECKMATE = WHITE_WINS_CHECKMATE | BLACK_WINS_CHECKMATE
    """Game ended in a checkmate."""

    DRAW_CONCLUSIVE = DRAW_STALEMATE | DRAW_INSUFFICIENT_MATERIAL | DRAW_SEVENTYFIVE_MOVES | DRAW_FIVEFOLD_REPETITION
    """Game ended in a draw, forced by the rules of the game."""

    DRAW_CLAIMED = DRAW_FIFTY_MOVES | DRAW_THREEFOLD_REPETITION
    """Game ended in a draw, claimed by one of the players."""

    DISQUALIFICATION = WHITE_WINS_DISQUALIFICATION | BLACK_WINS_DQ_INVALID_MOVE
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


_termination_to_outcome = {
    chess.Termination.SEVENTYFIVE_MOVES: Outcome.DRAW_SEVENTYFIVE_MOVES,
    chess.Termination.FIVEFOLD_REPETITION: Outcome.DRAW_FIVEFOLD_REPETITION,
    chess.Termination.INSUFFICIENT_MATERIAL: Outcome.DRAW_INSUFFICIENT_MATERIAL,
    chess.Termination.STALEMATE: Outcome.DRAW_STALEMATE,
}


class Game(TypedDict):
    """Outcome, details and statistics of a played game."""

    moves: SANMoveList
    """Moves played in the game."""

    num_opening_moves: int
    """The number of moves that were provided as an opening sequence."""

    outcome: Outcome

    # TODO: Doesn't work with TypedDict, use simple function
    # @staticmethod
    # def summary(games: List[Game]):
    #     """Returns statistics on a collection of games."""

    #     # Count canonical game outcomes.
    #     outcome_counts = {o: 0 for o in Outcome}
    #     for game in games:
    #         for outcome in Outcome:
    #             if game.outcome & outcome:
    #                 outcome_counts[outcome] += 1

    #     # Length of games.
    #     num_moves = [len(g.moves) for g in games]
    #     mean, std = np.mean(num_moves), np.std(num_moves)

    #     # Retries.
    #     # num_retries_black = [r for r in g.retries[1::2] for g in games]
    #     num_retries_black = [r for g in games for r in g.retries[1::2]]
    #     num_retries_white = [r for g in games for r in g.retries[::2]]
    #     num_retries_black_mean, num_retries_black_std = np.mean(num_retries_black), np.std(num_retries_black)
    #     num_retries_white_mean, num_retries_white_std = np.mean(num_retries_white), np.std(num_retries_white)
    #     num_retries = [r for g in games for r in g.retries]
    #     num_retries_mean, num_retries_std = np.mean(num_retries), np.std(num_retries)

    #     # Plot number of retries.
    #     return {
    #         "outcome_counts": outcome_counts,
    #         "num_moves_mean": mean,
    #         "num_moves_std": std,
    #         "num_retries_mean": num_retries_mean,
    #         "num_retries_std": num_retries_std,
    #         "num_retries_black_mean": num_retries_black_mean,
    #         "num_retries_black_std": num_retries_black_std,
    #         "num_retries_white_mean": num_retries_white_mean,
    #         "num_retries_white_std": num_retries_white_std,
    #     }


@enum.unique
class PlayerSignal(enum.Enum):
    """Signal communicated by players alternative to their move."""

    # TODO: Signale auf Seite der Player definieren, aber hier auflisten

    ABANDONED_GAME = enum.auto()
    """GUIPlayer: Closed window."""

    DIVERGED_FROM_PRESET = enum.auto()
    """Preset player: moves pushed to the movestack diverge from the preset."""

    NO_LEGAL_MOVES = enum.auto()
    """Random player: no legal moves available. Should not happen."""

    CONTEXT_OVERFLOW = enum.auto()
    """Transformer: context size exceeded."""

    CANT_CONSTRUCT_MOVE = enum.auto()
    """Transformer: couldn't synthesize a valid move."""


class SANPlayer(Protocol):
    """Chess player interface."""

    # Parameters with which player was initialized. Used to annotate
    # evaluations and performance measurements.
    # TODO: info() Methode:
    #       - name (testen, dass Namen eindeutig sind)
    info = {}

    @abstractmethod
    def __init__(self, movelist: SANMoveList = SANMoveList(())):
        """
        Initializes player with an optional movelist.
        """

    @abstractmethod
    def push_moves(self, movelist: SANMoveList):
        # TODO: Sollte alles TAN sein hier, nicht SAN
        """
        Pushes moves to the player's internal move stack. The moves are
        guaranteed to be valid.
        """

    @abstractmethod
    def suggest_move(self) -> PlayerSignal | SANMove:
        """
        Suggests moves in SAN format. Must not modify move stack.

        # TODO: Festlegen, wie n interpretiert werden soll.
        #       - Maximale Anzahl? Minimale? Müssen Züge unterschiedlich sein?
        """

    @abstractmethod
    def reset(self, movelist: SANMoveList = SANMoveList(())):
        """
        Resets the player's internal state followed by an initialization with
        the given movelist.
        """


class RandomPlayer(SANPlayer):
    """Plays random valid moves, optionally generating invalid moves."""

    board: chess.Board
    rng: random.Random

    def __init__(
        self,
        movelist: SANMoveList = SANMoveList(()),
        *,
        rng: random.Random = random.Random(),
    ):
        self.rng = rng
        self.reset(movelist)

    def push_moves(self, movelist: SANMoveList):
        for move in movelist:
            self.board.push_san(move)

    def suggest_move(self) -> Literal[PlayerSignal.NO_LEGAL_MOVES] | SANMove:
        legal_moves = list(self.board.legal_moves)
        if len(legal_moves) == 0:
            return PlayerSignal.NO_LEGAL_MOVES

        move_uci = self.rng.choice(legal_moves)
        move_san = uci_to_san(move_uci, self.board)
        return move_san

    def reset(self, movelist: SANMoveList = SANMoveList(())):
        self.board = chess.Board()
        self.push_moves(movelist)


class PresetPlayer(SANPlayer):
    """
    Picks moves from a preset sequence. Used for testing. Aborts game via
    signal if pushed moves diverge from or exceed preset sequence.
    """

    movelist: SANMoveList
    move_idx: int
    diverged_from_preset: bool

    def __init__(
        self,
        movelist: SANMoveList = SANMoveList(()),
    ):
        self.reset(movelist)

    def push_moves(self, movelist: SANMoveList):
        if self.movelist[self.move_idx : self.move_idx + len(movelist)] != movelist:
            self.diverged_from_preset = True
            return
        self.move_idx += len(movelist)

    def suggest_move(self) -> Literal[PlayerSignal.DIVERGED_FROM_PRESET] | SANMove:
        if self.diverged_from_preset or self.move_idx >= len(self.movelist):
            return PlayerSignal.DIVERGED_FROM_PRESET
        return self.movelist[self.move_idx]

    def reset(self, movelist: SANMoveList = SANMoveList(())):
        self.movelist = movelist
        self.move_idx = 0
        self.diverged_from_preset = False


class GUIPlayer(SANPlayer):
    """Gets moves via a GUI."""

    def __init__(self, movelist: Collection[str] = (), *, board_sz=800):
        # Tkinter setup.
        self.root = tk.Tk()
        self.root.title("Chess GUI")
        self.root.bind("<Escape>", self.close)
        self.board_sz = board_sz
        self.canvas = tk.Canvas(self.root, width=self.board_sz, height=self.board_sz)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)

        # We need to keep a reference to the image, otherwise the variable will
        # be garbage collected and the image memory will be freed, resulting in
        # the canvas being empty.See https://stackoverflow.com/questions/2223985
        self.tk_img = None

        self.sig: PlayerSignal | SANMove | None = None
        self.input_waiting = tk.BooleanVar()

        self.board = chess.Board()
        self.selected_square = None

        self.push_moves(movelist)

        self.info = {f"{movelist=}".split("=")[0]: deepcopy(movelist)}

    def push_moves(self, movelist: Collection[str]):
        for move in movelist:
            self.board.push_san(move)
        self.draw_board()

    def suggest_move(self) -> PlayerSignal | SANMove:
        self.root.wait_variable(self.input_waiting)
        if self.sig is not None:
            return self.sig
        raise NotImplementedError("this code path should never be reached")

    def close(self, _event):
        self.sig = PlayerSignal.ABANDONED_GAME
        self.root.destroy()
        self.input_waiting.set(True)

    def draw_board(self, **kwargs):
        # Draw the chess board using SVG. We have to convert to PGN first.
        svg_data = chess.svg.board(board=self.board, coordinates=False, **kwargs)
        png_data = cairosvg.svg2png(bytestring=svg_data, output_width=self.board_sz, output_height=self.board_sz)
        if not isinstance(png_data, bytes):
            raise ValueError("couldn't create SVG")
        img = Image.open(io.BytesIO(png_data))
        img = img.resize((self.board_sz, self.board_sz), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        self.root.update_idletasks()

    def on_click(self, event):
        square_sz = self.board_sz // 8
        file = max("a", min(chr((event.x // square_sz) + ord("a")), "h"))
        rank = max(1, min(8 - (event.y // square_sz), 8))
        square = f"{file}{rank}"

        if self.selected_square is None:
            self.selected_square = square
            fill = {chess.SQUARES[chess.parse_square(square)]: "#cc0000cc"}
        else:
            if self.selected_square == square:
                pass
            else:
                movename_uci = self.selected_square + square
                move_uci = chess.Move.from_uci(movename_uci)
                if move_uci in self.board.legal_moves:
                    # Push the move to the other player:
                    # Hack to generate a move in san notation:
                    # this will produce a continuation for black: '11...Rg8'
                    # or a move for white: '3. Qd3'. This has to be done
                    # before pushing to the board.
                    # TODO: Move this to a helper
                    variation = self.board.variation_san([move_uci])
                    movepos = variation.rfind(".")
                    move_san: SANMove
                    if variation[movepos + 1] == " ":
                        move_san = SANMove(variation[movepos + 2 :])
                    else:
                        move_san = SANMove(variation[movepos + 1 :])

                    move: TANMove = strip(move_san)

                    self.sig = move
                    self.input_waiting.set(True)

            self.selected_square = None
            fill = {}

        self.draw_board(fill=fill)

    def reset(self, movelist: Collection[str] = ()):
        pass
        # TODO:


def play_game(
    white: SANPlayer,
    black: SANPlayer | None = None,
    *,
    opening_moves: SANMoveList = SANMoveList(()),
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
        for move in opening_moves:
            board.push_san(move)
    except ValueError:
        return {
            "moves": opening_moves,
            "num_opening_moves": len(opening_moves),
            "outcome": Outcome.ABORT_INVALID_OPENING,
        }

    for p in players:
        p.push_moves(opening_moves)

    moves = deepcopy(list(opening_moves))
    while True:
        # Check if game has ended.
        outcome = board.outcome()
        if outcome is not None:
            return {
                "moves": SANMoveList(moves),
                "num_opening_moves": len(opening_moves),
                "outcome": Outcome.from_outcome(outcome),
            }

        # Get a move suggestions, validate and push it.
        players_idx = board.turn & (len(players) - 1)  # picks the correct player index
        response = players[players_idx].suggest_move()
        if isinstance(response, SANMove):
            move = response
            if not is_valid(move, board):
                outcome = Outcome.BLACK_WINS_DQ_INVALID_MOVE if board.turn else Outcome.WHITE_WINS_DISQUALIFICATION
                return {
                    "moves": SANMoveList(moves),
                    "num_opening_moves": len(opening_moves),
                    "outcome": outcome,
                }
            else:
                moves.append(move)
                for p in players:
                    p.push_moves(SANMoveList([move]))
        else:
            # Handle signals.
            # TODO: move signal handling to function
            signal = response
            outcome = outcome_from_signal(signal, board.turn)
            return {
                "moves": SANMoveList(moves),
                "num_opening_moves": len(opening_moves),
                "outcome": outcome,
            }


def get_outcome(san_movelist: SANMoveList) -> Outcome:
    """Returns a game's outcome by playing it out by a PresetPlayer.
    Inconclusive games will result in a resignation signal."""
    # TODO: Was, wenn Spiel nicht terminiert? play_game kann hier so nicht verwendet werden.
    game = play_game(PresetPlayer(), opening_moves=san_movelist)
    return game["outcome"]


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
        Outcome.DRAW_INSUFFICIENT_MATERIAL: ["e4 d5 exd5 Qxd5 Nc3 Qxg2 Qh5 Qxh1 Qxh7 Qxh2 Qxh8 Qxg1 Qxg8 Nc6 d3 Qxf1 Kd2 Qxc1 Ke2 Qxa1 Kd2 Qxa2 Kc1 Qxb2 Kd1 Qxc3 Ke2 Qxc2 Ke1 Qxd3 f3 Qxf3 Qxg7 Qg2 Qxf8 Kd7 Qxf7 Qg3 Kd2 b6 Qf8 Ke6 Qxc8 Kf6 Qxa8 Kg7 Qxa7 Kf8 Qxb6 Kg7 Qxc7 Kg8 Qxc6 Kg7 Qd6 Kg8 Qxe7 Qg7 Qxg7 Kxg7".split(" ")],
        Outcome.DRAW_FIVEFOLD_REPETITION: [
            # https://lichess.org/UzKC66ai
            "e4 d6 d4 Nf6 Nc3 g5 Bxg5 h6 Bxf6 exf6 Nf3 Nc6 Bb5 Bd7 Qd2 Qe7 O-O-O O-O-O Rhe1 a6 Ba4 b5 Bb3 Na5 Nd5 Nxb3 cxb3 Qe6 Qc3 c6 Kb1 Kb7 Rc1 Rc8 Qa5 cxd5 exd5 Qxd5 Rxc8 Bxc8 Rc1 Bg7 Qc7 Ka8 Qa5 Bf5 Ka1 Rc8 Qxa6 Kb8 Qb6 Ka8 Qa6 Kb8 Qb6 Ka8 Qa6 Kb8 Qb6 Ka8 Qa6 Kb8 Qb6 Ka8 Qa6".split(" ")
        ],
        Outcome.DRAW_SEVENTYFIVE_MOVES: ["Nc3 c5 b4 cxb4 e4 Na6 Bd3 f6 f4 b6 Bb2 Bb7 Ba3 Rb8 Bxb4 e6 Bxa6 d6 Ba3 Qc7 Bxb7 Qd8 Ba6 g5 Bb5+ Qd7 g4 h6 Nf3 Kf7 fxg5 Rd8 gxh6 Ke8 Ng5 Bxh6 Nh7 Qxb5 Qe2 Bxd2+ Kxd2 Kd7 Bc1 Kc8 g5 Ne7 Qd1 a6 Na4 Qd3+ Kxd3 Nd5 Bb2 Rde8 Rc1 Kd8 Rg1 Kc7 Qg4 Ne7 Qd1 Kd7 Nc3 Ra8 Qd2 Rxh7 Rg3 Rc8 Nb1 Ng8 Rh1 Ke7 c4 Rf8 h4 a5 Qd1 fxg5 h5 Rhf7 Rgg1 Rh7 Be5 Nh6 Kc2 Rf6 Nc3 Ke8 Na4 Rhf7 Rxg5 Rg7 Rf5 Kd7 Rg5 Nf5 Nb2 Kd8 Qd2 Kd7 Rh3 Nd4+ Kd3 Nc6 Qc2 Rf1 Re3 Rh1 Qf2 a4 Bd4 Nxd4 Rf3 Nf5 Qh2 Rg1 Qh3 Ra1 Rfg3 Rb1 Rg1 Nd4 Qe3 d5 a3 Nc2 Qxb6 Ne3 h6 Re7 Kxe3 Kc8 Rxd5 Rc7 Qxe6+ Kb7 Qc8+ Rxc8 Rg2 Rb8 Rd6 Rg8 Kd4 Rg7 Kd5 Kb8 Rg4 Rg5+ Kc6 Rg6 Re6 Re1 Rf4 Rf6 Rfxf6 Rh1 Rf7 Rh5 Ref6 Rd5 Nd3 Ra5 e5 Rc5+ Kd6 Rxe5 Rh7 Re7 Rf1 Re8 Re7 Rg8 Nb4 Rg7 Rf5 Rg1 Rb7+ Ka8 Nc2 Ra1 Re7 Rh1 Ne3 Kb8 Nc2 Ka8 Na1 Rh2 Kd7 Kb7 Rf8 Rh4 Rfe8 Re4 Kd6+ Kb6 Rg7 Rf4 Re1 Ka6 Rd1 Rf5 Rd4 Rd5+ cxd5 Ka5 Rf4 Ka6 Rg8 Ka7 Rc4 Kb7 Rh4 Ka6 Rhg4 Kb5 Rd4 Ka5 Re8 Ka6 Rg4 Ka5 Rf8 Ka6 Rf3 Kb5 Ke7 Ka6 Rb3 Ka5 Ke8 Ka6 Nc2 Ka7 Rh4 Ka6 Kf8 axb3 Re4 bxc2 Ke8 Ka5 Rg4 c1=R Rg6 Rh1 Kd7 Rd1 Kc8 Rb1 a4 Rb6 h7 Rd6 Rf6 Kxa4 Rf2 Kb5 Rd2 Kc4 Re2 Rf6 Re7 Rd6 Rd7 Kd3 Ra7 Rf6 Rb7 Rf7 Rb4 Rxh7 Kb8 Rh5 Rc4 Kd2 Rf4 Rh3 Ra4 Ke3 Re4+ Kd3 Re6 Rg3 Re7 Rh3 d6 Rh7 Rb7 Re7 Rb6 Kd2 Rc6 Rg7 Rc4 Kd3 Rc6 Ke4 Rc3 Rg3 Rxg3 Kd4 Rg4+ Kc5 Rg2 Kb5 Rg5+ Kb4 Rh5 Ka3 Ka7 Kb3 Re5 Ka4 Rc5 Ka3 Rh5 Kb2 Ka8 Kb3 Rc5 Ka2 Rc4 Kb1 Rh4 Ka2 Rh5 Kb3 Kb7 Ka3 Kb8 Ka2 Rh6 Kb3 Rh5 Kc3 Rh8 Kc2 Rh1 Kb3 d7 Kc3 d8=N Kb3 Rg1 Ka4 Rh1 Kb3 Kc8 Ka4 Kb8 Kb3 Re1 Ka4 Ne6 Kb4 Nf4 Ka5 Ng2 Kb6 Re3 Kb5 Kb7 Kc5 Rc3+ Kb5 Ra3 Kc5 Rb3 Kd6 Rb4 Ke7 Kb6 Kf6 Rb1 Ke6 Kb7 Kf5 Re1 Kf6 Kc6 Kg6 Re5 Kf6 Re4 Kg7 Re5 Kh6 Re3 Kg6 Rd3 Kf6 Ra3 Kg7 Nf4 Kh8 Ng2 Kg8 Ne3 Kf8 Kc7 Kf7 Kc8 Kg8 Kc7 Kf8 Ra8+ Kg7 Rb8 Kh7 Kc8 Kg7 Rb4 Kf7 Rb6 Ke7 Kb7 Ke8 Ka7 Ke7 Rf6 Kd7 Rg6 Ke7 Rg1 Ke8 Kb7 Ke7 Rg6 Kd7 Ka8 Kd8 Rb6 Ke8 Rb8+ Kd7 Rh8 Kc7 Rh2 Kc6 Nf1 Kc5 Ng3 Kc4 Nh1 Kb4 Kb7 Ka4 Kb8 Ka5 Ka7 Ka4 Rd2 Kb3 Ka6 Kb4 Rb2+ Kc5 Rf2 Kc4 Rc2+ Kb3 Rc4 Kb2 Ra4 Kb3 Ra5 Kb2 Nf2 Kc3 Rd5 Kb3 Rd2 Kc4 Rd8 Kc3 Re8 Kb2 Re1 Kc3 Rg1 Kb2 Rd1 Kb3 Rd2 Ka4 Rd5 Kb3 Ka5 Kc3 Re5 Kb2 Rg5".split(" ")],
    }
)


# TODO: test
def is_valid(
    move: SANMove | TANMove | str,
    board: chess.Board,
) -> bool:
    """
    Returns true if the move is valid, given a position. The `board` object is
    not modified.

    :param move: move in standard algebraic notation
    :param board: chess.board object representing the position
    """

    # Pychess doesn't offer a method to check the validity of a move in SAN
    # notation, so we have to call push_san() directly and look for exceptions.
    # However, we must not modify the move stack and thus do this on a copy of
    # the board.
    board_cp = deepcopy(board)
    try:
        board_cp.push_san(move)
    except ValueError:
        return False
    return True


def tan_moveline_from_gameline(tan_gameline: str) -> str:
    tan_gameline = tan_gameline.rstrip()
    if tan_gameline.endswith(TAN_EOG_CHARS):
        return tan_gameline[:-2]  # strip eog char and trailing whitespace
    return tan_gameline


def strip(move: SANMove) -> TANMove:
    """
    Strips annotations from a SANMove.

    Returns a TANMove.
    """
    result = move.rstrip(SAN_MOVE_ANNOTATION_POSTFIX)
    return TANMove(SANMove(result))


def uci_to_san(move_uci: chess.Move, board: chess.Board) -> SANMove:
    # Hack to generate a move in san notation: this will produce a continuation
    # for black, e.g. '11...Rg8' or a move for white: '3. Qd3'.
    variation = board.variation_san([move_uci])
    movepos = variation.rfind(".")
    if variation[movepos + 1] == " ":
        move = SANMove(variation[movepos + 2 :])
    else:
        move = SANMove(variation[movepos + 1 :])

    return SANMove(move)


def outcome_from_signal(signal: PlayerSignal, turn: chess.Color) -> Outcome:
    if signal == PlayerSignal.DIVERGED_FROM_PRESET:
        if turn == chess.WHITE:
            return Outcome.BLACK_WINS_RESIGNATION_DIVERGED_FROM_PRESET
        return Outcome.WHITE_WINS_RESIGNATION_DIVERGED_FROM_PRESET

    if signal == PlayerSignal.CONTEXT_OVERFLOW:
        if turn == chess.WHITE:
            return Outcome.BLACK_WINS_RESIGNATION_CONTEXT_OVERFLOW
        return Outcome.WHITE_WINS_RESIGNATION_CONTEXT_OVERFLOW

    if signal == PlayerSignal.ABANDONED_GAME:
        if turn == chess.WHITE:
            return Outcome.BLACK_WINS_RESIGNATION_ABANDONED_GAME
        return Outcome.WHITE_WINS_RESIGNATION_ABANDONED_GAME

    if signal == PlayerSignal.CANT_CONSTRUCT_MOVE:
        if turn == chess.WHITE:
            return Outcome.BLACK_WINS_RESIGNATION_CANT_CONSTRUCT_MOVE
        return Outcome.WHITE_WINS_RESIGNATION_CANT_CONSTRUCT_MOVE

    if signal == PlayerSignal.NO_LEGAL_MOVES:
        if turn == chess.WHITE:
            return Outcome.BLACK_WINS_RESIGNATION_NO_LEGAL_MOVES
        return Outcome.WHITE_WINS_RESIGNATION_NO_LEGAL_MOVES

    raise NotImplementedError(f"Unknown signal: {signal}")


# TODO: SAN.. komplett rausschmeißen, falls möglich
