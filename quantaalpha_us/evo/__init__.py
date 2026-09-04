"""An evolutionary factor-mining loop, and the controls that say whether it helps.

The repo's existing miner is one shot: one prompt, N expressions, score them,
done. Everything in the LLM-for-alpha literature that reports a gain reports it
from a LOOP -- FunSearch, AlphaEvolve, ReEvo, LLaMEA, QuantaAlpha -- so the
honest question is not "can a model write a factor" but "does iterating with
feedback beat asking once, and if it does, is the gain the loop or the model?".

Answering that needs the loop AND its controls to run through identical gates,
identical fitness, identical archive rules and identical windows. That is what
this package is: a small MAP-Elites loop over islands, with a genetic-programming
arm that swaps only the proposal step, so a difference between the two is
attributable to the model rather than to the scaffolding.

Windows are fixed before anything runs (see `config.Windows`): fit 2000-2013,
validation 2014-2017 for the early-stop rule and the two threshold choices, and
a holdout of 2018-2025 that one script reads once at the very end. No prompt
ever contains a number computed on validation or holdout, or a date after
2013-12-31; `tests/test_evo_prompt_audit.py` scans every prompt on disk for
both.
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
