import pytest

from src.tan_chess.common import (
    tan_moveline_from_gameline,
    is_valid_moveline,
)

TAN_GAMELINES = [
    "e4 e6 d4 b6 a3 Bb7 Nc3 Nh6 Bxh6 gxh6 Be2 Qg5 Bg4 h5 Nf3 Qg6 Nh4 Qg5 Bxh5 Qxh4 Qf3 Kd8 Qxf7 Nc6 Qe8 W",
    "e4 g6 d4 d6 Nf3 c6 h3 Nf6 Bg5 Nxe4 Qe2 Bf5 Nbd2 Qa5 c3 Nxd2 Bxd2 Nd7 b4 Qa3 Ng5 h5 Qc4 d5 Qe2 Qb2 Qd1 Bc2 Qc1 Qxc1 Rxc1 Ba4 Bd3 Nb6 O-O Nc4 Bxc4 dxc4 Bf4 Bh6 Rfe1 O-O Rxe7 Rae8 Rxb7 f6 Ne6 Rxe6 Bxh6 Rf7 Rb8 Kh7 Bf4 g5 Bd2 Re2 Be1 Rfe7 Kf1 Bc2 Rc8 Bd3 Rxc6 Rc2 Kg1 Rxc1 Rxf6 h4 g4 Rexe1 Kg2 Be4 f3 Rc2 S",
    "e4 e5 Nf3 Nc6 Bb5 Nge7 Nc3 h6 Nd5 a6 Ba4 b5 Bb3 Ng6 c4 b4 Ba4 Nd4 O-O c6 Ne3 a5 Nxd4 exd4 Nf5 Ba6 Nxd4 Bxc4 Nxc6 Bxf1 Nxd8 Bxg2 Nxf7 Bxe4 Qe2 Kxf7 Qxe4 Kg8 Qxa8 Kh7 Qe4 d5 Qxd5 Be7 d3 Rf8 Bb3 Nf4 Qe4 Ng6 Be3 Bh4 Rf1 Bf6 Bd4 Bh4 f4 Bd8 f5 Ne7 f6 Ng6 fxg7 Rxf1 Kxf1 h5 g8=Q Kh6 Qexg6 W",
    "e4 c5 f4 d5 exd5 Qxd5 Nc3 Qd8 Bc4 Bf5 d3 a6 g4 Bd7 a4 e6 Bd2 Bc6 Nf3 Bxf3 Qxf3 Qh4 Qg3 Qxg3 hxg3 Nc6 O-O-O O-O-O f5 Ne5 fxe6 Nxc4 dxc4 fxe6 Rde1 Bd6 Bf4 Bxf4 gxf4 Nh6 g5 Nf5 Rxe6 Rd4 Rf1 Rxc4 Re5 g6 Kd2 Rd8 Kc1 Rd7 Nd5 Rd6 Ne7 Nxe7 Rxe7 Rd7 Rxd7 Kxd7 b3 Re4 Kb2 Ke6 Kc3 Kf5 Rh1 Re7 Rf1 Re4 Rh1 Rxf4 Rxh7 Kxg5 Rxb7 Rf6 Rc7 Kf4 Rxc5 g5 b4 g4 Rc4 Kf3 Rc5 Rg6 Rf5 Kg2 b5 axb5 axb5 g3 Kb4 Kh1 Rd5 g2 Rd1 g1=Q Rxg1 Kxg1 c4 Kf2 c5 Ke3 b6 Kd4 b7 Rg1 Kb5 Rb1 Kc6 Rb4 Kc7 Kxc5 b8=Q Rxb8 Kxb8 U",
    "b3 e5 Bb2 e4 d3 Nf6 Nh3 d5 dxe4 Nxe4 Nf4 Qh4 g3 Bc5 f3 Bf2 S",
]


class Test_tan_moveline_from_gameline:
    assert tan_moveline_from_gameline.__name__ == "tan_moveline_from_gameline"

    @pytest.mark.parametrize("tan_gameline", TAN_GAMELINES)
    def test(self, tan_gameline):
        tan_moveline = tan_moveline_from_gameline(tan_gameline)
        assert is_valid_moveline(tan_moveline)

    @pytest.mark.parametrize("tan_gameline", TAN_GAMELINES)
    def test_input_moveline(self, tan_gameline):
        tan_moveline = tan_moveline_from_gameline(tan_gameline)
        assert tan_moveline == tan_moveline_from_gameline(tan_moveline)
