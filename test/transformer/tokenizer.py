from src.transformer.tokenizer import (
    encode_gameline,
    START_OF_GAME_TOKEN_ID,
    END_OF_GAME_TOKEN_ID,
)
from src.tan_chess.common import movelist_to_moveline
from src.tan_chess.game import Outcome

from ..assets.games import conclusive_games


class Test_encode_gameline:

    assert encode_gameline.__name__ == "encode_gameline"

    def test_default(self):
        moveline = movelist_to_moveline(conclusive_games[Outcome.WHITE_WINS_CHECKMATE][0])
        gameline = moveline + " W"
        encd_gameline = encode_gameline(gameline)

        assert len(encd_gameline.shape) == 1
        assert len(encd_gameline) == len(gameline) + 1
        assert encd_gameline[0] == START_OF_GAME_TOKEN_ID
        assert encd_gameline[-2] == END_OF_GAME_TOKEN_ID
