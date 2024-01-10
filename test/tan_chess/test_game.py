import inspect
from itertools import combinations
from enum import Enum
import importlib
import pkgutil

import pytest

from src.tan_chess import Outcome, TANPlayer, get_outcome, play_game, PresetPlayer
from ..assets.games import conclusive_games


def import_all_modules(package_name: str):
    """
    Recusively imports all modules from a root package.
    """

    package = importlib.import_module(package_name)
    for _, module_name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_module_name = f"{package_name}.{module_name}"
        importlib.import_module(full_module_name)
        if is_pkg:
            import_all_modules(full_module_name)


@pytest.fixture
def tan_player_subclasses():
    """
    Returns all subclasses of TANPlayer. Imports all modules in the src/
    directory, so that all subclasses are found.
    """

    import_all_modules("src")
    return TANPlayer.__subclasses__()


class TestPlayerSpecificResignationOutcome:
    def test_tan_player_subclasses_have_unique_capitalized_names(self, tan_player_subclasses):
        """
        Asserts that all subclasses of TANPlayer have unique names when
        capitalized. This is a necessary condition for the dynamic generation of
        the Outcome enum, whose items referring to resignation look like

            {WHITE,BLACK}_WINS_RESIGNATION_<CAPITALIZED_PLAYER_CLASS_NAME>_<RESIGNATION_REASON>
        """

        for a, b in combinations(tan_player_subclasses, 2):
            capitalized_names_are_different = a.__name__.upper() != b.__name__.upper()
            errmsg = f"{inspect.getfile(a)}:{a.__name__} and {inspect.getfile(b)}:{b.__name__} have matching names if capitalized"
            assert capitalized_names_are_different, errmsg

    def test_tan_player_subclass_resignation_signal_contract(self, tan_player_subclasses):
        """
        Tests whether all subclasses of TANPlayer implement the
        ResignationReason StrEnum class member. This is necessary, as we extend
        the Outcome Enum by the different resignation reason, specific to each
        player class.
        """

        for subclass in tan_player_subclasses:
            has_resignation_field = TANPlayer.ResignationReason.__name__ in vars(subclass)
            resignation_field_is_enum = issubclass(subclass.ResignationReason, Enum)
            errmsg = f"{inspect.getfile(subclass)}:{subclass.__name__} does not correctly implement a 'ResignationReason' Enum"
            assert has_resignation_field and resignation_field_is_enum, errmsg

    def test_tan_player_subclass_resignation_signals_are_registered_in_outcome(self, tan_player_subclasses):
        resignation_outcomes = {o.name for o in Outcome}
        for subclass in tan_player_subclasses:
            for resignation_reason in (r.name for r in subclass.ResignationReason):
                for side in ("WHITE", "BLACK"):
                    player_specific_resignation_outcome = f"{side}_WINS_RESIGNATION_{subclass.__name__.upper()}_{resignation_reason}"
                    player_specific_resignation_outcome_exists = player_specific_resignation_outcome in resignation_outcomes
                    assert player_specific_resignation_outcome_exists, f"{player_specific_resignation_outcome} is missing from the Outcome enum"


class Test_get_outcome:
    assert get_outcome.__name__ == "get_outcome"

    def test_happy_path(self):
        fools_mate = conclusive_games[Outcome.WHITE_WINS_CHECKMATE][0]
        outcome = get_outcome(fools_mate)
        assert outcome == Outcome.WHITE_WINS_CHECKMATE

    def test_incomplete_game(self):
        fools_mate = conclusive_games[Outcome.WHITE_WINS_CHECKMATE][0]
        outcome = get_outcome(fools_mate[:-1])
        assert outcome == Outcome.INCOMPLETE_GAME

    def test_invalid_moves(self):
        fools_mate = conclusive_games[Outcome.WHITE_WINS_CHECKMATE][0]
        outcome = get_outcome(fools_mate[:-1] + ["a8"])
        assert outcome == Outcome.ABORT_INVALID_OPENING


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
