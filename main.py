from src import lichess_db
import argparse
import os

def extract(database_file, output):
    if output is None:
        output = database_file + "-extracted.tan"

    # Create subdirectories, if necessary.
    os.makedirs(os.path.dirname(output), mode=0o755, exist_ok=True)

    i = 0
    with open(output, "w") as outfile:
        for movetext in lichess_db.extract_movetexts(database_file):
            outfile.write(movetext + "\n")
            if i % 1000 == 0:
                print("\r" * 100 + f"Extracted {i+1} games to '{output}'... ", end="")
            i += 1
    print(f"done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser")

    extract_subparser = subparsers.add_parser("extract", help="Extract trimmed movetext of games from a lichess games database file.")
    extract_subparser.add_argument("database_file", help="Database of games in PGN format (zst-encoded).")
    extract_subparser.add_argument("-o", "--output", default=None, dest="output", help="Output file containing the movetexts. By default, appends the suffix '-extracted.tan' to the database file's path.")
    args = parser.parse_args()

    if args.subparser == "extract":
        extract(args.database_file, args.output)