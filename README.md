# Milestones

1. Using character-level tokenization, train a transformer on TAN movelines of conclusive games (checkmate, draw by fivefold repetition or by the 75-move rule). The goal is to learn to play valid moves.
    - Alternative approach: Instead of training on sequences of algebraic notation, use the symmetric UCI format to represent moves (ca. 64^2 possible moves == embeddings).