"""Functionality to transform lichess chess games database files avaliable at
https://database.lichess.org/"""

import zstandard
import io
import re

def extract_movetexts(database):
    """Extracts the movetexts in trimmed algebraic notation from an zst-encoded
    database of chess games in PGN format.
    
    This function returns a generator which yields one full game's movetext per
    output. The outputs do not contain a trailing newline."""
    with open(database, 'rb') as fh:
        dctx = zstandard.ZstdDecompressor(max_window_size=2147483648)
        stream_reader = dctx.stream_reader(fh)
        text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
        for line in text_stream:
            line = line.rstrip()
            if line.startswith("1. "):
                yield trim_movetext(line)

def trim_movetext(movetext):
    """Trims movetext in algebraic notation by removing move number
    indicators and common annotations.

    Expects a string representing a full game's movetext in algebraic notation."""
    # Remove move indices. In addition to regular move indices, some moves
    # are annotated with engine evaluations and use use a triple period to
    # designate the continuation of the movetext:
    #
    #   1. b4 { [%eval -0.46] } 1... d5 { [%eval -0.44] } 2. Bb2 ...
    movetext = re.sub(r"[1-9][0-9]*\.+\s?", "", movetext)
    
    # Remove annotations in brackets {}.
    movetext = re.sub(r"\{[^}]*\}\s?", "", movetext)
    
    # Remove move quality annotations such as blunders or brilliant moves.
    movetext = re.sub(r"[?!]", "", movetext)
        
    return movetext