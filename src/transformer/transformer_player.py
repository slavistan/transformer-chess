from __future__ import annotations

from enum import Enum, auto
import torch
import chess

from src.tan_chess import (
    TAN_MAX_MOVE_LEN,
    TANMove,
    TANMoveList,
    TANPlayer,
    is_valid_move,
)
from .vanilla_transformer import Model
from .tools import (
    PADDING_IDX,
    WHITESPACE_IDX,
    encode_moveline_as_np8,
    decode_moveline_tensor,
)


class TransformerPlayer(TANPlayer):
    board: chess.Board
    model: Model
    model_context_sz: int
    model_device: str
    movetensor: torch.Tensor
    write_idx: int
    context_overflow: bool
    num_tries_until_valid: int

    def __init__(
        self,
        model: Model,
        movelist=(),
        num_tries_until_valid=1,
    ):
        self.model = model
        self.model_context_sz = self.model.init_params["context_sz"]
        self.model_device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.num_tries_until_valid = num_tries_until_valid
        self.push_moves(movelist)
        self.reset()

    def push_moves(
        self,
        movelist: TANMoveList,
    ) -> TransformerPlayer:
        if self.context_overflow:
            return self

        for m in movelist:
            encd = encode_moveline_as_np8(m + " ")
            num_tokens = len(encd)

            # Silently stop adding moves once we've reached the end of the
            # tensor. Resignation will follow.
            if self.write_idx + num_tokens >= len(self.movetensor):
                self.context_overflow = True
                return self

            self.movetensor[self.write_idx : self.write_idx + num_tokens] = torch.from_numpy(encd)
            self.write_idx += num_tokens
            self.board.push_san(m)

        return self

    def suggest_move(
        self,
    ) -> TransformerPlayer.ResignationReason | TANMove:
        if self.context_overflow:
            return TransformerPlayer.ResignationReason.CONTEXT_OVERFLOW

        for _ in range(self.num_tries_until_valid):
            movetensor_buffer = self.movetensor.clone()
            write_idx_buffer = self.write_idx

            while True:
                token = self.model.generate(movetensor_buffer[:write_idx_buffer], num_tokens=1).item()

                # Abort once we exceed the maximum number of characters a move
                # in TAN format can consist of.
                num_generated_tokens = write_idx_buffer - self.write_idx
                if num_generated_tokens >= TAN_MAX_MOVE_LEN:
                    break

                # Skip over padding tokens.
                # TODO: Padding tokens erscheinen in den Daten nach dem letzten
                #       Zug eines Spiels, sind also Teil eines sinnvollen Outputs.
                #       Vielleicht sollte ich padding tokens und whitespaces gleich
                #       behandeln.
                #       Alternativvorschlag: Spiele werden direkt mit einem extra Leerzeichen
                #                            am Ende der Zugsequenz kodiert, sodass Paddingtokens
                #                            niemals nach einem Zug erscheinen.
                if token == PADDING_IDX:
                    continue

                # Break if whitespace is returned. Whitespace is not part of
                # the returned move. However, continue if no tokens have been
                # generated yet.
                if token == WHITESPACE_IDX:
                    if num_generated_tokens == 0:
                        continue
                    break

                movetensor_buffer[write_idx_buffer] = token
                write_idx_buffer += 1

            # Decode generated move.
            move = decode_moveline_tensor(movetensor_buffer[self.write_idx : write_idx_buffer])
            if is_valid_move(move, self.board):
                return move

        return TransformerPlayer.ResignationReason.CANT_CONSTRUCT_VALID_MOVE

    def reset(
        self,
    ) -> TransformerPlayer:
        self.board = chess.Board()
        self.movetensor = torch.zeros(
            (self.model_context_sz,),
            dtype=torch.uint8,
            device=self.model_device,
            requires_grad=False,
        )
        self.write_idx = 1  # write pointer; index 0 is reserved for padding
        self.context_overflow = False

        return self

    class ResignationReason(Enum):
        CONTEXT_OVERFLOW = auto()
        """Preset context size of transformer exceeded by game tokens"""

        CANT_CONSTRUCT_VALID_MOVE = auto()
        """Failed to produce a valid move after a set amount of attempts"""
