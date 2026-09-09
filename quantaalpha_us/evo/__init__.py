"""An evolutionary factor-mining loop, and the controls that say whether it helps.

The repo's existing miner is one shot: one prompt, N expressions, score them,
done. Iterative search motivates testing whether feedback beats asking once,
and whether any gain comes from the loop or the proposal model.

Answering that needs the loop AND its controls to run through identical gates,
identical fitness, identical archive rules and identical windows. That is what
this package is: a small MAP-Elites loop over islands, with a genetic-programming
arm that swaps only the proposal step, so a difference between the two is
attributable to the model rather than to the scaffolding.

Windows are fixed before anything runs (see `config.Windows`): fit 2000-2013,
validation 2014-2017 for early stopping, threshold choices and final ridge selection, and
a holdout of 2018-2025 that one script reads once at the very end. No prompt
contains an explicitly labelled validation or holdout score, or a date after
2013-12-31. Prompt scanning cannot detect historical forward-label leakage;
the repaired scorer separately enforces outcome-end cutoffs.
"""

from quantaalpha_us.evo.config import (
    ISLANDS,
    EvoConfig,
    Gates,
    FitnessWeights,
    Island,
    Schedule,
    Windows,
)

__all__ = [
    "ISLANDS", "EvoConfig", "Gates", "FitnessWeights", "Island", "Schedule", "Windows",
]
