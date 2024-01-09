import pytest

from src.tan_chess.common import TANPlayer

def test_tan_player_subclass_resignation_signal_contract():
    for subclass in TANPlayer.__subclasses__():
        assert hasattr(subclass, "ResignationReason"), f"{subclass.__name__} does not implement the ResignationReason enum"
