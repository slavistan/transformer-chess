import pytest

from src.tan_chess import PresetPlayer, TANMoveList

from ..assets.games import conclusive_games


class Test_PresetPlayer:
    assert PresetPlayer.__name__ == "PresetPlayer"

    @pytest.mark.parametrize("movelist", [g for games in conclusive_games.values() for g in games])
    def test_happy_path(self, movelist: TANMoveList):
        p = PresetPlayer(movelist)
        for move in movelist:
            response = p.suggest_move()
            assert response == move
            p.push_moves([move])

    def test_divert_from_preset(self):
        movelist = ["e4", "d5", "a4"]
        p = PresetPlayer(movelist)

        # Assert that we get the propert signal even if we're repeatedly pushing unexpected moves.
        for _ in range(8):
            p.push_moves(["a4"])
            sig = p.suggest_move()
            assert sig == PresetPlayer.ResignationReason.DIVERGED_FROM_PRESET

    def test_exceed_preset(self):
        movelist = ["e4", "d5", "a4"]
        p = PresetPlayer(movelist)

        for move in movelist:
            p.push_moves([move])

        # Sends out of moves signal once the movelist is exceeded.
        sig = p.suggest_move()
        assert sig == PresetPlayer.ResignationReason.EXCEEDED_PRESET
