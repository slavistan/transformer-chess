import pytest

from src.tan_chess import play_game, PresetPlayer
from ..assets.games import conclusive_games


class Test_play_game:
    assert play_game.__name__ == "play_game"

    @pytest.mark.parametrize("movelist,outcome", [(g, outcome) for outcome, games in conclusive_games.items() for g in games])
    def test_happy_path_1player_without_opening(self, movelist, outcome):
        game_1p = play_game(PresetPlayer(movelist))
        assert game_1p["outcome"] == outcome

    @pytest.mark.parametrize("movelist,outcome,opening", [(g, outcome, g[:i]) for outcome, games in conclusive_games.items() for g in games for i in range(len(g))])
    def test_happy_path_1player_with_opening(self, movelist, outcome, opening):
        game_1p = play_game(PresetPlayer(movelist), opening=opening)
        assert game_1p["outcome"] == outcome

    @pytest.mark.parametrize("movelist,outcome", [(g, outcome) for outcome, games in conclusive_games.items() for g in games])
    def test_happy_path_2players_without_opening(self, movelist, outcome):
        game_1p = play_game(PresetPlayer(movelist), PresetPlayer(movelist))
        assert game_1p["outcome"] == outcome

    @pytest.mark.parametrize("movelist,outcome,opening", [(g, outcome, g[:i]) for outcome, games in conclusive_games.items() for g in games for i in range(len(g))])
    def test_happy_path_2players_with_opening(self, movelist, outcome, opening):
        game_1p = play_game(PresetPlayer(movelist), PresetPlayer(movelist), opening=opening)
        assert game_1p["outcome"] == outcome
