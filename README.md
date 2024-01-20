# Milestones

# TODO: Implement pre-fetching data loading and train on big dataset of conclusive games
# TODO: Test game understanding by ability to find legal moves in position of very few legal moves
# TODO: Test game understanding by ability to one-move checkmates
# TODO: Test game understanding by ability to promote pawns
# TODO: Try different sampling methods (nucleus sampling)

1. Using character-level tokenization, train a transformer on TAN movelines of conclusive games (checkmate, draw by fivefold repetition or by the 75-move rule). The goal is to learn to play valid moves.
    - Alternative approach: Instead of training on sequences of algebraic notation, use the symmetric UCI format to represent moves (ca. 64^2 possible moves == embeddings).

# TODO: Abdeckung legaler Züge des Transformers evaluieren
#       - Alle legalen Züge ausgeben und via "Beamsearch" vergleichen
# TODO: SANPlayer.suggest_moves() könnte Züge mit Konfidenz zurückgeben `(sig, {"a4", 0.3, "Qxf1": 0.7})`
# TODO: Proxytask 'Erkennen des Gewinners am Ende eines definiten Spiels' könnte eigentliche Aufgabe (Schachspielen) verbessern

Idee
- RandomPlayers zur Erzeugung von Trainingsspielen verwenden
- Schachbrett mit Anzahl benötigter Wiederholungen versehen
- Schachbrett mit Wahrscheinlichkeitsmasse der legalen Züge versehen
- Headsize automatisch aus Anzahl Heads und Embeddingdim bestimmen
- Trainingsdaten dynamisch aus Positionen erzeugen, mit denen der Transformer gemäß der Wahrscheinlichkeitmasse legaler Züge Probleme hat. Ausgehend von diesen Position via RandomPlayer für jeden legalen Zug ein Spiel erzeugen.