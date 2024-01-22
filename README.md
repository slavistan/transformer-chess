# Roadmap

## V1.0

- Rework db_utils to use new multiprocessing approach.
- Check if training games are, in fact, conclusive.
- TransformerPlayer
    - GUIPlayer: Visualize the TransformerPlayer's legal move prob of every position
    - GUIPlayer: Visualize the TransformerPlayer's top n moves
    - Eval: Add an evaluation for the probability mass of legal moves over a set of games.
    - Eval: Rework one-move checkmate puzzles to determine the outcome using the probability of the target move instead of playing out the puzzle.
    - Eval: Heatmap of legal move prob (color) vs length of opening (x) vs number of legal moves (y)
    - Eval: Evaluate the ability to promote pawns
    - Eval: Evaluate the ability to predict the correct outcome of a conclusive game
    - Eval: ELO-Rating ermitteln (siehe [SO](https://chess.stackexchange.com/questions/12790/how-to-measure-strength-of-my-own-chess-engine))
- Transformer
    - Check allocation of memory on the correct device. Something's not quite right here, especially when loading from file via .load().
    - Synthetically augment real-world game data according to how well a position is understood by the transformer, gauged by the probability mass assigned to legal moves.

## V2.0

- Combine multiple lichess databases and remove duplicate games.
- Transformer: Rework transformer to remove unaligned head sizes. Head size is determined from the number of heads and the embedding dimension.
- Try different sampling methods (top-k, top-p)
- Fine-tuning with games from high-performance computer chess engines.