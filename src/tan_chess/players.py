from __future__ import annotations

import io
from enum import Enum, auto
import tkinter as tk
import random

import chess
import chess.svg
import cairosvg
from PIL import Image, ImageTk

from .common import (
    TANMove,
    TANMoveList,
    TANPlayer,
    uci_to_tan,
)


class PresetPlayer(TANPlayer):
    """
    Picks moves from a preset sequence. Used for testing. Aborts game via
    signal if pushed moves diverge from or exceed preset sequence.
    """

    movelist: TANMoveList
    move_idx: int
    diverged_from_preset: bool

    def __init__(
        self,
        movelist: TANMoveList,
    ):
        self.reset(movelist)

    def push_moves(
        self,
        movelist: TANMoveList,
    ) -> PresetPlayer:
        for m in movelist:
            if self.move_idx >= len(self.movelist):
                return self
            if self.movelist[self.move_idx] != m:
                self.diverged_from_preset = True
                return self
            self.move_idx += 1
        return self

    def suggest_move(
        self,
    ) -> PresetPlayer.ResignationReason | TANMove:
        if self.diverged_from_preset:
            return PresetPlayer.ResignationReason.DIVERGED_FROM_PRESET
        if self.move_idx >= len(self.movelist):
            return PresetPlayer.ResignationReason.EXCEEDED_PRESET

        return self.movelist[self.move_idx]

    def reset(self, movelist: TANMoveList = ()) -> PresetPlayer:
        self.movelist = movelist
        self.move_idx = 0
        self.diverged_from_preset = False
        return self

    class ResignationReason(Enum):
        DIVERGED_FROM_PRESET = auto()
        """Diverged from preset list of moves by pushing a mismatching move"""

        EXCEEDED_PRESET = auto()
        """Exceeded preset movelist"""


class RandomPlayer(TANPlayer):
    """Plays random valid moves."""

    board: chess.Board
    rng: random.Random

    def __init__(
        self,
        rng=random.Random(),
    ):
        self.rng = rng
        self.reset()

    def push_moves(
        self,
        movelist: TANMoveList,
    ) -> RandomPlayer:
        for move in movelist:
            self.board.push_san(move)
        return self

    def suggest_move(
        self,
    ) -> RandomPlayer.ResignationReason | TANMove:
        legal_moves = list(self.board.legal_moves)
        if len(legal_moves) == 0:
            return RandomPlayer.ResignationReason.NO_LEGAL_MOVES

        move_uci = self.rng.choice(legal_moves)
        move_san = uci_to_tan(move_uci, self.board)
        return move_san

    def reset(
        self,
    ) -> RandomPlayer:
        self.board = chess.Board()
        return self

    class ResignationReason(Enum):
        NO_LEGAL_MOVES = auto()
        """No legal moves to pick from"""


class GUIPlayer(TANPlayer):
    """Gets moves via a GUI."""

    def __init__(
        self,
        *,
        board_sz=800,
    ):
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

        self.sig: GUIPlayer.ResignationReason | TANMove | None = None  # buffer for response of player
        self.input_waiting = tk.BooleanVar()

        self.board = chess.Board()
        self.selected_square = None

    def push_moves(
        self,
        movelist: TANMoveList,
    ) -> GUIPlayer:
        for move in movelist:
            self.board.push_san(move)
        self.draw_board()
        return self

    def suggest_move(
        self,
    ) -> GUIPlayer.ResignationReason | TANMove:
        self.root.wait_variable(self.input_waiting)
        if self.sig is not None:
            return self.sig
        raise NotImplementedError("this code path should never be reached")

    def close(
        self,
        _event,
    ):
        self.sig = GUIPlayer.ResignationReason.ABANDONED_GAME
        self.root.destroy()
        self.input_waiting.set(True)

    def draw_board(
        self,
        **kwargs,
    ):
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

    def reset(
        self,
    ) -> GUIPlayer:
        """TODO: implement"""
        return self

    def on_click(
        self,
        event,
    ):
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
                    move = uci_to_tan(move_uci, self.board)
                    self.sig = move
                    self.input_waiting.set(True)

            self.selected_square = None
            fill = {}

        self.draw_board(fill=fill)

    class ResignationReason(Enum):
        ABANDONED_GAME = auto()
        """GUI Player abandoned game before it finished, e.g. by closing the window"""
