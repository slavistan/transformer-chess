"""Tests for db_utils.py"""

from src import db_utils, tan_chess
import multiprocessing as mp
import os
from pathlib import Path


def test_san_gameline_to_tan():
    """Tests san_game_to_tan()."""
    table = (
        (
            r"1. e4 { [%clk 0:01:00] } 1... c5 { [%clk 0:01:00] } 2. Nf3 { [%clk 0:00:59] } 2... Nc6 { [%clk 0:01:00] } 3. c3 { [%clk 0:00:58] } 3... e5 { [%clk 0:00:59] } 4. d4 { [%clk 0:00:58] } 4... cxd4 { [%clk 0:00:59] } 5. cxd4 { [%clk 0:00:58] } 5... exd4 { [%clk 0:00:59] } 6. Nxd4 { [%clk 0:00:57] } 6... Nf6 { [%clk 0:00:59] } 7. Nxc6 { [%clk 0:00:56] } 7... bxc6 { [%clk 0:00:59] } 8. Nc3 { [%clk 0:00:56] } 8... Bb4 { [%clk 0:00:58] } 9. f3 { [%clk 0:00:54] } 9... O-O { [%clk 0:00:57] } 10. Bd2 { [%clk 0:00:53] } 10... a5 { [%clk 0:00:57] } 11. a3 { [%clk 0:00:52] } 11... Be7 { [%clk 0:00:56] } 12. Bc4 { [%clk 0:00:51] } 12... Ba6 { [%clk 0:00:56] } 13. Qe2 { [%clk 0:00:50] } 13... Bxc4 { [%clk 0:00:54] } 14. Qxc4 { [%clk 0:00:50] } 14... Qc7 { [%clk 0:00:54] } 15. O-O { [%clk 0:00:49] } 15... Rac8 { [%clk 0:00:53] } 16. Qd3 { [%clk 0:00:48] } 16... Rfd8 { [%clk 0:00:52] } 17. Bg5 { [%clk 0:00:45] } 17... h6 { [%clk 0:00:51] } 18. Bxf6 { [%clk 0:00:44] } 18... Bxf6 { [%clk 0:00:51] } 19. Rad1 { [%clk 0:00:43] } 19... Qb6+ { [%clk 0:00:49] } 20. Kh1 { [%clk 0:00:37] } 20... Qxb2 { [%clk 0:00:48] } 21. Ne2 { [%clk 0:00:37] } 21... c5 { [%clk 0:00:45] } 22. Rd2 { [%clk 0:00:37] } 22... Qe5 { [%clk 0:00:43] } 23. Nc1 { [%clk 0:00:34] } 23... c4 { [%clk 0:00:42] } 24. Qe2 { [%clk 0:00:34] } 24... c3 { [%clk 0:00:41] } 25. Na2 { [%clk 0:00:34] } 25... cxd2 { [%clk 0:00:40] } 26. Nc3 { [%clk 0:00:33] } 26... Qxc3 { [%clk 0:00:39] } 27. Qf2 { [%clk 0:00:32] } 27... Qc1 { [%clk 0:00:37] } 28. Qg3 { [%clk 0:00:32] } 28... Qxf1# { [%clk 0:00:36] } 0-1",
            r"e4 c5 Nf3 Nc6 c3 e5 d4 cxd4 cxd4 exd4 Nxd4 Nf6 Nxc6 bxc6 Nc3 Bb4 f3 O-O Bd2 a5 a3 Be7 Bc4 Ba6 Qe2 Bxc4 Qxc4 Qc7 O-O Rac8 Qd3 Rfd8 Bg5 h6 Bxf6 Bxf6 Rad1 Qb6 Kh1 Qxb2 Ne2 c5 Rd2 Qe5 Nc1 c4 Qe2 c3 Na2 cxd2 Nc3 Qxc3 Qf2 Qc1 Qg3 Qxf1 S",
        ),
        (
            r"1. c4 e5 2. Nf3 Nc6 3. e4 Bc5 4. Nc3 Nf6 5. a3 O-O *",
            r"c4 e5 Nf3 Nc6 e4 Bc5 Nc3 Nf6 a3 O-O",
        ),
    )

    for test in table:
        # Test handling of no newline characters, linux and windows EOL.
        want, have = test[1], db_utils.pgn_gameline_to_tan(test[0])
        assert have == want

        want, have = test[1], db_utils.pgn_gameline_to_tan(test[0] + "\n")
        assert have == want

        want, have = test[1], db_utils.pgn_gameline_to_tan(test[0] + "\r\n")
        assert have == want


def test_splitfn_lines(tmp_path: Path):
    content = "Is this the real life?\n" + "Is this just fantasy?\n" + "Malmö\r\n" + "\n"
    p = tmp_path / "hello.txt"
    p.write_text(content, encoding="utf-8")

    lines = db_utils.splitfn_lines_sequential(str(p))
    pairs = [(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)]

    print(content)

    assert pairs[0][0] == 0
    assert pairs[0][1] == 23
    assert pairs[1][0] == 23
    assert pairs[1][1] == 22
    assert pairs[2][0] == 45
    assert pairs[2][1] == 8
    assert pairs[3][0] == 53
    assert pairs[3][1] == 1
    assert len(pairs) == 4  # final newline to EOF does not constitute a line


def test_make_collectfn_write(tmp_path: Path):
    results = [["abc\n", "def", "ghi\n"], ["abc\n", "def", "ghi\n"], ["abc\n", "def", "ghi\n"]]
    out_path = tmp_path / "foo"
    writefn = db_utils.make_writefn(str(out_path), newline=True)
    writefn(results)

    assert os.path.exists(str(out_path))
    with open(str(out_path), "r") as f:
        lines = f.readlines()
        assert lines == ["abc\n", "def\n", "ghi\n"] * 3

    out_path = tmp_path / "bar"
    writefn = db_utils.make_writefn(str(out_path), newline=False)
    writefn(results)

    assert os.path.exists(str(out_path))
    with open(str(out_path), "r") as f:
        lines = f.readlines()
        assert lines == ["abc\n", "defghi\n"] * 3


# TODO: das testet das Filtern von Spielen; auslagern
#       parallel_process muss separat, mglw. mit Dummy fns getestet werden.
def test_parallel_process(tmp_path: Path):
    out_path = tmp_path / "foo"

    # Pseudo filter games (matching every game outcome). Check if all games are
    # returned.
    for num_workers in [1, 4, mp.cpu_count()]:
        split_fn = db_utils.splitfn_lines_sequential
        process_fn = db_utils.processfn_filter_by_outcome
        process_fn_extra_args = (tan_chess.Outcome.CHECKMATE | ~tan_chess.Outcome.CHECKMATE,)
        collect_fn = db_utils.make_writefn(str(out_path))
        db_utils.parallel_process(
            "./data/example.tan",
            split_fn,
            process_fn,
            collect_fn,
            process_fn_extra_args=process_fn_extra_args,
            num_workers=num_workers,
            quiet=True,
        )

        with open(str(out_path), "r") as new_f, open("./data/example.tan", "r") as f:
            games_new = sorted(list(new_f.readlines()))
            games = sorted(list(f.readlines()))

            assert games == games_new
