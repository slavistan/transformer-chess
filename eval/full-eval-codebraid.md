---
title: "Performance Evaluation Report"
codebraid:
    jupyter: true
---

<!--
Visualizes a `EvalResult`.

Live Reload via entr:
    entr -cs "DATA_JSON=result.json PYTHONPATH=\"$(realpath .)\" codebraid pandoc --no-cache --overwrite --from markdown --to pdf -o cb.pdf ./eval/full-eval-codebraid.md" <<<./eval/full-eval-codebraid.md
-->

```{.python .cb-run}
import os
import json
from src.tan_chess import performance
from plotnine import labs

data_json = os.environ["DATA_JSON"]
with open(data_json, "r") as f:
    data = json.load(f)
```

**Visualzation of evaluation results `data_json`{.python .cb-nb}**.

# Games

## Against Random Player as White

```{.python .cb-run show=rich_output}
plot = performance.plot_outcome_hist(data["vs_random_as_white"])
plot
```

```{.python .cb-run show=rich_output}
plot = performance.plot_game_len_hist(data["vs_random_as_white"])
plot
```

## Against Random Player as Black

```{.python .cb-run show=rich_output}
plot = performance.plot_outcome_hist(data["vs_random_as_black"])
plot
```

```{.python .cb-run show=rich_output}
plot = performance.plot_game_len_hist(data["vs_random_as_black"])
plot
```

## Against Self

```{.python .cb-run show=rich_output}
plot = performance.plot_outcome_hist(data["vs_self"])
plot
```

```{.python .cb-run show=rich_output}
plot = performance.plot_game_len_hist(data["vs_self"])
plot
```

# Puzzles

## One-Move Checkmates

```{.python .cb-run show=rich_output}
plot = performance.plot_puzzle_likelihood_vs_num_of_legal_moves(data["one_move_checkmate_puzzles"])
plot
```

```{.python .cb-run show=rich_output}
plot = performance.plot_puzzle_likelihood_vs_len_of_opening(data["one_move_checkmate_puzzles"])
plot
```

```{.python .cb-run show=rich_output}
plot = performance.plot_puzzle_len_of_opening_vs_num_legal_moves(data["one_move_checkmate_puzzles"])
plot
```
