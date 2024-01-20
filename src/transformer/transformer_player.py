from __future__ import annotations

from copy import deepcopy
from enum import Enum, auto
import json

import torch
import chess


from src.tan_chess import (
    TAN_MAX_MOVE_LEN,
    TANMove,
    TANMoveList,
    TANPlayer,
    is_valid_move,
    full_eval,
    make_report,
    one_move_puzzle_from_tan,
    is_valid_movelist,
    is_conclusive_movelist,
    get_legal_moves,
)
from .vanilla_transformer import VanillaTransformer
from .tools import (
    PADDING_TOKEN_ID,
    WHITESPACE_TOKEN_ID,
    START_OF_GAME_TOKEN_ID,
    END_OF_GAME_TOKEN_ID,
    encode_move,
    decode_move,
    encode_moveline,
)


class TransformerPlayer(TANPlayer):
    model: VanillaTransformer
    model_context_sz: int
    model_device: str
    movetensor: torch.Tensor
    num_tries_until_valid: int

    board = chess.Board()
    write_idx = 1
    context_overflow = False

    def __init__(
        self,
        model: VanillaTransformer,
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
            encoded_move = encode_move(m + " ")
            num_tokens = len(encoded_move)

            # Silently stop adding moves once we've reached the end of the
            # tensor. Resignation will follow.
            if self.write_idx + num_tokens >= len(self.movetensor):
                self.context_overflow = True
                return self

            self.movetensor[self.write_idx : self.write_idx + num_tokens] = encoded_move
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
                # Break once we exceed the maximum number of characters a move
                # in TAN format can consist of.
                num_generated_tokens = write_idx_buffer - self.write_idx
                if num_generated_tokens >= TAN_MAX_MOVE_LEN:
                    break

                token = self.model.generate(movetensor_buffer[:write_idx_buffer], num_tokens=1).item()

                # Padding tokens or start-of-game tokens are always invalid in the
                # context of this method.
                if token in (PADDING_TOKEN_ID, START_OF_GAME_TOKEN_ID):
                    continue

                if token == END_OF_GAME_TOKEN_ID:
                    pass

                # Break if whitespace is returned. Whitespace is not part of
                # the returned move. However, continue if no tokens have been
                # generated yet.
                #
                # TODO: Handle end-of-game tokens correctly.
                #       If an end-of-game token is returned, we check if the
                #       suggested move does, in fact, result in a game ending
                #       result. In that case, we return the (valid) move
                #       suggestion. Otherwise, this counts as an invalid move.
                if token in (WHITESPACE_TOKEN_ID, END_OF_GAME_TOKEN_ID):
                    if num_generated_tokens == 0:
                        continue
                    break

                movetensor_buffer[write_idx_buffer] = token
                write_idx_buffer += 1

            # Decode generated move.
            move = decode_move(movetensor_buffer[self.write_idx : write_idx_buffer])
            if is_valid_move(move, self.board):
                return move

        return TransformerPlayer.ResignationReason.CANT_CONSTRUCT_VALID_MOVE

    def reset(
        self,
    ) -> TransformerPlayer:
        self.board = chess.Board()
        self.movetensor = torch.full(
            (self.model_context_sz,),
            PADDING_TOKEN_ID,
            dtype=torch.uint8,
            device=self.model_device,
            requires_grad=False,
        )
        self.movetensor[0] = START_OF_GAME_TOKEN_ID
        self.write_idx = 1  # write pointer; index 0 is reserved for start-of-game token
        self.context_overflow = False

        return self

    class ResignationReason(Enum):
        CONTEXT_OVERFLOW = auto()
        """Preset context size of transformer exceeded by game tokens"""

        CANT_CONSTRUCT_VALID_MOVE = auto()
        """Failed to produce a valid move after a set amount of attempts"""

    def prob_of_valid_moves(
        self,
        opening: TANMoveList = (),
    ) -> float:
        assert is_valid_movelist(opening), "opening movelist is invalid"
        assert not is_conclusive_movelist(opening), "opening movelist must not be conclusive"

        board = chess.Board()
        for m in opening:
            board.push_san(m)

        encoded_continuations = []
        for tan_move in get_legal_moves(board):
            board_cp = deepcopy(board)
            board_cp.push_san(tan_move)

            # Encode move
            t = torch.empty((len(tan_move) + 1,), dtype=torch.uint8)
            t[: len(tan_move)] = encode_move(tan_move)

            # Depending on whether the move concludes the game we add the
            # end-of-game token or the move separator token.
            t[-1] = WHITESPACE_TOKEN_ID if board_cp.outcome() is None else END_OF_GAME_TOKEN_ID
            encoded_continuations.append(t)

        prefix = encode_moveline(" ".join(opening) + " ")
        prob = 0.0
        for cont in encoded_continuations:
            prob_cont = self.model.prob_of_continuation(prefix, cont)
            prob += prob_cont

        return prob


def full_eval_transformer(
    pth_file: str,
    data_output_path: str,
    report_output_path: str,
    *,
    num_random=64,
    num_self=64,
    num_puzzles=256,
    num_puzzle_attempts=64,
    num_workers=1,
    num_tries_until_valid=16,
):
    # TODO: machine-independent way of storing puzzles
    puzzles = list(
        one_move_puzzle_from_tan(
            "./data/2309-checkmate.tan",
            num_games=num_puzzles,
        )
    )

    m = VanillaTransformer.load(pth_file).to("cpu")
    player = TransformerPlayer(m, num_tries_until_valid=num_tries_until_valid)
    eval_results = full_eval(
        player,
        puzzles,
        num_random=num_random,
        num_self=num_self,
        num_puzzle_attempts=num_puzzle_attempts,
        num_workers=num_workers,
    )

    with open(data_output_path, "w") as f:
        json.dump(eval_results, f, indent=4, default=str)

    return make_report(data_output_path, report_output_path)
