# LLMStrat

LLMStrat is a US equities research and execution stack for daily S&P 500 alpha mining. It builds a point-in-time universe, maintains market data, evaluates candidate signals with walk-forward controls, and can route approved portfolios into Alpaca paper or live trading with explicit risk checks.

## Provenance

This project descends from **QuantaAlpha** — [QuantaAlpha/QuantaAlpha](https://github.com/QuantaAlpha/QuantaAlpha), MIT License, paper: *QuantaAlpha: An Evolutionary Framework for LLM-Driven Alpha Mining* ([arXiv:2602.07085](https://arxiv.org/abs/2602.07085)). QuantaAlpha targets the China A-share market; this is a rewrite for US equities.

**Kept from QuantaAlpha:**

- The **staged research pipeline** concept: ideate, express, sanitise, evaluate, gate — with the LLM confined to the ideation stage and every stage after it mechanical and auditable.
- The **formulaic-alpha DSL** as the LLM's output contract, rather than free-form generated code. 22 of this repo's 32 operator names (`TS_MEAN`, `RANK`, `DELAY`, `TS_CORR`, `ZSCORE`, …) also appear in upstream's function library, though most are the common Alpha101/Qlib vocabulary predating both projects.
- The **LLM-ideation framing**: a frontier model is a hypothesis generator whose output is worthless until it survives an evaluation harness it cannot influence.

**Rebuilt here, with no upstream counterpart:** the data layer (CRSP via WRDS, replacing Qlib/A-share); the point-in-time S&P 500 membership filter joined on PERMNO rather than ticker; rank-space deduplication of candidate signals; the out-of-sample holdout with frozen in-sample sign; the sanitizer's identifier and arity checks; the Claude Code LLM backend; and the append-only, content-addressed experiment trace.

**Not carried over:** upstream's evolutionary search itself — the trajectory-level mutation and crossover operators that are the paper's actual contribution. This repo does single-shot ideation plus selection. It inherits QuantaAlpha's scaffolding, not its algorithm.

At the code level the two share nothing: a file-by-file comparison of all 27 modules in `quantaalpha_us/` against all 159 upstream modules found no file above 0.17 similarity, and the four files that share a basename with an upstream file share only imports and `@dataclass` decorators. The full table, the method, and its caveats are in **[docs/PROVENANCE.md](docs/PROVENANCE.md)**.

### On comparing IC against the paper

Upstream's headline result is an **IC of 0.1501 on CSI 300** (GPT-5.2 backbone, ARR 27.75%, MDD 7.98%). This repo measures mean ICs of **0.003–0.011** on the S&P 500. Those numbers are not comparable, and the gap should not be read as either a validation or a failure of this port:

- **Different market.** Upstream's headline is CSI 300. US large-cap is the most heavily arbitraged equity universe in the world; daily cross-sectional ICs there are structurally smaller than in A-shares.
- **The paper reports no US IC.** QuantaAlpha does test zero-shot transfer of CSI-300-mined factors onto the S&P 500, and reports it as successful — roughly 137% cumulative excess return over the 2022–2025 test window (Figure 1). But that is a portfolio-level cumulative excess return under a TopkDropout strategy with China-calibrated transaction costs, not an information coefficient. There is no published upstream S&P 500 IC to benchmark against.
- **Different search.** The evolutionary mutation/crossover loop that produces upstream's headline number is not implemented here.

The honest statement is the one already made under [Known limits](#known-limits): ICs of 0.003–0.011 are weak against the 0.02–0.05 of a decent published factor, and the large t-statistics sit below the noise floor of a search this size — random expressions from the same grammar reach 9 to 12. Factor quality, not tooling, is this repo's binding constraint.

### Citation

If you use this work, please cite the upstream paper:

```bibtex
@misc{han2026quantaalphaevolutionaryframeworkllmdriven,
      title={QuantaAlpha: An Evolutionary Framework for LLM-Driven Alpha Mining},
      author={Jun Han and Shuo Zhang and Wei Li and Zhi Yang and Yifan Dong and Tu Hu and Jialuo Yuan and Xiaomin Yu and Yumo Zhu and Fangqi Lou and Xin Guo and Zhaowei Liu and Tianyi Jiang and Ruichuan An and Jingping Liu and Biao Wu and Rongze Chen and Kunyi Wang and Yifan Wang and Sen Hu and Xinbing Kong and Liwen Zhang and Ronghao Chen and Huacan Wang},
      year={2026},
      eprint={2602.07085},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST},
      url={https://arxiv.org/abs/2602.07085},
}
```

## Repository layout

- `configs/`: research, paper, live, and LLM runtime configuration
- `quantaalpha_us/`: reusable package code for data, factors, backtests, and execution
- `scripts/`: entry points for universe construction, ingestion, research, signal generation, and trading
- `tests/`: regression coverage for data quality, factor mining, risk controls, and walk-forward validation
- `bootstrap.sh`: one-command environment setup for a fresh clone

## Setup

```bash
./bootstrap.sh
source .venv/bin/activate
```

## CLI usage

Build or refresh the S&P 500 membership table:

```bash
python scripts/sp500_build_membership.py --help
```

Backfill or refresh daily market data:

```bash
python scripts/sp500_ingest_daily.py --help
```

Run the core walk-forward research loop:

```bash
python scripts/sp500_run_research.py --config configs/backtest_sp500_research.yaml
```

## Methodology

The system is organized as a staged research pipeline rather than a single notebook. It first constructs a point-in-time investable universe, ingests and validates daily OHLCV data, and computes baseline cross-sectional features. Those signals feed a constrained long-only portfolio construction step, which is then evaluated in walk-forward windows with explicit costs, turnover limits, and promotion gates. Frontier LLMs are used only for bounded factor ideation and are surrounded by validation, budget controls, and expression sanitization before any candidate reaches research or trading.

## Output

Typical runs produce:

- point-in-time universe and reference tables under `data/us_equities`
- processed bars and coverage artifacts for daily research
- signal files and walk-forward backtest outputs
- factor-mining candidate files and validation summaries
- broker-facing rebalance intents for paper or live deployment

## Mined-factor research loop (2026-08)

The loop from mining to backtest is now closed end to end:

1. **Evaluator** (`quantaalpha_us/factors/expression_evaluator.py`): sanitized expression strings such as `TS_MEAN($close, 21) / (TS_STD($close, 21) + 1e-8)` are parsed with a strict AST whitelist (second guard behind the sanitizer) and evaluated into date x symbol signal panels. All time-series operators use strict windows, so warm-up periods are NaN rather than biased.
2. **Research scoring** (`quantaalpha_us/factors/factor_research.py` and `scripts/sp500_score_mined_factors.py`): candidates are scored by daily cross-sectional Spearman IC against forward open-to-open returns under the repo's execution convention, with an IC t-statistic across days and a signal-autocorrelation turnover proxy. Selection is greedy by |t-stat| with a pairwise signal-correlation cap, because string-level dedup cannot catch an LLM restating one idea five ways. That cap is applied in **rank space**, matching the rank IC selection ranks on: pooled Pearson on raw values is not invariant to a monotone cross-sectional transform, so `CS_RANK(X)` and `X` — identical orderings, identical IC — measured 0.21 against each other and both entered a set billed as uncorrelated. In rank space that pair is 1.00.
3. **Backtest integration**: `signals.mined_expressions` in the research config feeds selected expressions into `build_features`, where they are cross-sectionally ranked and averaged into the score next to the baseline factors, and the gate-6 stability measurement covers them automatically.

`configs/mined_factors_claude_2026-08.txt` ships a candidate set generated by Claude Fable 5 (Anthropic). These are candidates, not validated factors: every one must clear the IC scoring script on research-grade data and then the walk-forward promotion gates. A unit test guarantees the whole file sanitizes and evaluates.

## Auditable experiment trace

A mining loop that keeps only its winners cannot be audited. The rejected hypotheses are what show whether a search was disciplined or simply ran until something passed, and they are exactly what a leaderboard discards.

`quantaalpha_us/factors/experiment_trace.py` records every hypothesis considered — the expression, the reasoning that produced it, the scores it earned, and the verdict, failures included. It follows the auditable-trace design argued for in [arXiv 2604.26747](https://arxiv.org/abs/2604.26747), with two properties that make it evidence rather than decoration:

- **Append-only.** Records are JSON Lines, flushed and fsynced on write, never rewritten. There is no update or delete operation, because a trace that can be edited afterwards proves nothing about what was tried.
- **Content-addressed.** Each record carries a hash of the normalised expression, so the same idea proposed twice is detected across runs even when spelled differently. That is what lets the trace act as search memory rather than a log.

The trace is also **the multiple-testing denominator**. Running the fundamental candidate set:

```
verdicts: {'selected': 10, 'rejected_correlated': 1, 'rejected_score': 1, 'distinct_expressions': 12}
distinct hypotheses ever tried: 12 (use this as the DSR trial count, not 10)
```

A Deflated Sharpe computed against the survivors understates the search by exactly the number of rejections. Computed against the trace, it reflects what was actually tried — and across many runs the trace accumulates, so the denominator keeps growing the way an honest one should.

```bash
python scripts/sp500_score_mined_factors.py --bars <panel> --trace data/trace.jsonl
```

## Out-of-sample holdout

Factors are selected on **2000-2017** and reported on **2018-2025**, with the
in-sample choices frozen: which expressions, and which sign each was oriented
to. Refitting the sign out of sample would be the same overfitting one layer
down.

| candidate set | keeps sign OOS | median IC retention |
|---|---|---|
| Claude Fable 5 (20 candidates) | 7 / 8 | **29.5%** |
| Claude Sonnet 5 (54 candidates) | 10 / 10 | **52.8%** |
| fundamental (12 candidates) | 8 / 9 | **93.3%** |

The price-only sets lose most of their edge. On the Fable set, **one of eight
factors clears |t| > 2 out of sample**, against eight of eight in sample by
construction, and one flips sign outright. That is the factor-zoo result
measured on this repo's own factors rather than cited from a paper.

Signals are computed over the full panel and only the *IC dates* are
restricted. Every operator is backward-looking, so this leaks nothing, and
unlike slicing the bars it does not blank the holdout's first 252 days to
rolling-window warm-up.

`--train-end none` selects on the full sample and says so.

### This holdout is not a post-cutoff test

The panel ends **2025-12-31**, and every candidate set here was written by a
model trained on data running through 2025 and beyond. The 2018-2025 holdout is
therefore *inside* the generating model's training window. It is out of sample
for the selection step and in sample for the model that proposed the
expressions, which is a weaker claim than "out of sample" usually implies: a
model can propose a factor because it worked, without any of the reasoning that
would make it work again.

**No post-training-cutoff window exists in this repo today.** A real post-cutoff
test needs 2026 bars appended to the panel and the frozen expressions scored on
that window alone. Until that data is here, treat every retention number above
as an upper bound. The
[recent literature on LLM-generated alpha](#does-the-llm-add-anything) is
directly about this failure mode.

## Does the LLM add anything?

**No. On this repo's data, the LLM-generated factor sets do not beat expressions
drawn at random from the same grammar, and both are beaten by a 2015 published
factor set.** The strongest Sonnet candidate scores |t| = 11.64 in sample; five
seeds of random expression-generation, matched for candidate count and
structure, produce best |t| between 9.11 and 12.41. Sonnet sits inside that
range. Fable, at 9.82, sits below its median.

Reproduce the whole table with one command:

```bash
python scripts/sp500_run_baseline_comparison.py --bars data/us_equities/processed/daily_bars.parquet
```

| set | candidates | scorable | best \|t\| in-sample | E[max \|t\|] under iid null | selected | OOS median IC retention | sign held OOS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Claude Fable 5 | 20 | 20 | 9.82 | 2.17 | 8 | 29.5% | 7/8 |
| Claude Sonnet 5 | 54 | 53 | 11.64 | 2.53 | 10 | 52.8% | 10/10 |
| random-grammar-seed0 | 54 | 54 | 10.97 | 2.54 | 10 | 20.5% | 7/10 |
| random-grammar-seed1 | 54 | 54 | 9.80 | 2.54 | 10 | 66.6% | 8/10 |
| random-grammar-seed2 | 54 | 54 | 9.93 | 2.54 | 10 | 10.7% | 5/10 |
| random-grammar-seed3 | 54 | 54 | 12.41 | 2.54 | 10 | 28.2% | 7/10 |
| random-grammar-seed4 | 54 | 54 | 9.11 | 2.54 | 10 | 18.3% | 7/10 |
| Alpha101 (published) | 50 | 50 | 12.11 | 2.51 | 10 | 58.8% | 10/10 |

Every `best |t| in-sample` above is measured on the **2000-2017 selection window
(4,528 trading days)**, with 2018-2025 held back and reported separately in the
two right-hand columns. It is therefore not the same quantity as the 7.56 under
[Universe](#universe), which is a full-sample figure — comparing the two
directly would be comparing an 18-year window against a 26-year one.

Three things fall out of it.

**The random baseline is not weak, and neither is the LLM.** Random-grammar
retention ranges from 10.7% to 66.6% across seeds. Sonnet's 52.8% is inside
that spread, and seed 1 beats it. With five seeds the honest reading is that
retention is dominated by draw-to-draw variance, not by which process wrote the
expressions. A single random seed would have been enough to "prove" either
conclusion, which is exactly why there are five.

**The iid null is far too generous, and the empirical one is the real bar.**
E[max |t|] over N independent standard-normal draws is about 2.5 at N = 54. Every
set here clears 9. So `|t| = 8` is not evidence of anything on this panel: the
random-grammar row says a same-sized search over the same grammar reaches 9 to
12 routinely. This also reframes the |t| figures under
[Known limits](#known-limits): they are below the noise floor of a search this
size, not merely inflated by sample length.

*Corrected 2026-09.* An earlier version of this section attributed that gap to
daily IC autocorrelation, on the reasoning that the t-statistic assumes 4,500
independent days and does not get them. That explanation is measurably wrong.
Newey-West standard errors (Bartlett, 21 lags) across a 397-factor pool move the
median t-statistic by a factor of **1.03** and the pool maximum from 10.88 to
10.27; see [How many survive a factor-zoo
correction](#how-many-survive-a-factor-zoo-correction). Daily cross-sectional
rank ICs are close to serially uncorrelated here. The correct explanation is
that the random expressions are not noise: they are functions of price and
volume, so many of them are restatements of size, liquidity or short-term
reversal and carry genuine non-zero mean ICs. The iid null is a null of zero
mean, and these draws do not have zero mean.

**A multiple-testing correction does not rescue the model sets.** Applying
Harvey-Liu-Zhu, Benjamini-Hochberg, a Romano-Wolf stepdown and an empirical
random-grammar bar to the whole 397-factor pool, random expressions survive at a
HIGHER rate than the model's factors (41% against 28% under Romano-Wolf), and
Alpha101 at 54%. Among the survivors, 42% mention `$dollar_volume` and those
have a median holdout retention of -29%. The full table is in
[How many survive a factor-zoo
correction](#how-many-survive-a-factor-zoo-correction).

**Alpha101 wins.** A factor set published in 2015, transcribed mechanically, is
at least as good as both LLM sets on every column: highest in-sample |t| of the
non-random sets, highest retention, 10/10 signs held. Whatever the LLM is doing,
it is not beating a decade-old reference.

### The memorization test

For every LLM candidate, its highest correlation to any of the 50 transcribed
Alpha101 signals, computed on the same panel through the same evaluator. The
statistic is the **mean over days of the per-day cross-sectional rank
correlation** — the same within-day quantity the IC itself measures:

| set | candidates | max corr | median corr | share > 0.9 |
|---|---:|---:|---:|---:|
| Claude Fable 5 | 20 | 0.878 | 0.416 | **0%** |
| Claude Sonnet 5 | 53 | 0.950 | 0.412 | **4%** (2 of 53) |

A pooled correlation over all date-symbol pairs is reported alongside it in
`memorization_test.csv`. The two never differ by more than 0.011 here, so
nothing below turns on the choice.

Outright restatement is rare, and that is the more interesting result: the LLM
is mostly *not* reciting Alpha101, it is producing genuinely different
expressions that are no better than random ones. The two Sonnet candidates above
0.9 are near-duplicates of published alphas — the top match, at 0.95, is
`CS_RANK((($close - $open) / $open) - (($open - DELAY($close, 1)) / DELAY($close, 1)))`
against alpha033 — so a mining run that reports them as discoveries is
overcounting its own novelty, but only twice.

Correlation is one axis and it is not enough on its own: it misses a
restatement whose numbers happen to differ, and it flags coincidences.
[Is the model paraphrasing](#is-the-model-paraphrasing) adds the independent
structural axis, and finds a case correlation missed: a Sonnet candidate that
reuses a seven-node sub-expression of a published alpha whole.

**No paper in the LLM-for-alpha literature this repo builds on runs a control
like this.** QuantaAlpha (arXiv 2602.07085), ReEvo (NeurIPS 2024), FunSearch,
AlphaEvolve and LLaMEA all evaluate mined factors or heuristics against
benchmarks, and none of them checks whether the model reproduced something from
its training data rather than discovering it. Every one of them mines with a
model trained on the literature it is being tested against. That check is cheap
once a published reference set is transcribed onto the same panel, it is two
statistics rather than one, and it is the difference between "the model found
this" and "the model remembered this".

### What this does not show

This is a memorization and null-comparison test, not a post-cutoff test. Both
generating models were trained on data covering 2018-2025, so the holdout cannot
separate "the model reasoned well" from "the model remembered". The design of a
real test is fixed and waiting on data: append 2026 bars, score the already
frozen expressions on 2026 alone, and compare against the same random-grammar
seeds. See [This holdout is not a post-cutoff test](#this-holdout-is-not-a-post-cutoff-test).

The framing and the failure modes come from the 2025-26 work on LLM-generated
alpha: Look-Ahead-Bench ([arXiv 2601.13770](https://arxiv.org/abs/2601.13770)),
MemGuard-Alpha ([arXiv 2603.26797](https://arxiv.org/abs/2603.26797)), The Alpha
Illusion ([arXiv 2605.16895](https://arxiv.org/abs/2605.16895)) and Profit
Mirage ([arXiv 2510.07920](https://arxiv.org/abs/2510.07920)), which argue that
LLM-proposed signals are contaminated by training-window memorization and
frequently fail to beat naive baselines once the evaluation window clears the
cutoff. The table above is that argument reproduced on this repo's own factors.

### How the baselines are built

- **Random grammar** (`quantaalpha_us/factors/random_expressions.py`) samples
  from exactly the grammar the sanitizer accepts — the function set, arities and
  field names are read out of `ExpressionSanitizer.FUNCTION_ARITY`,
  `VARIADIC_MIN_ARITY` and `KNOWN_FIELDS` rather than restated, and a test fails
  if the two ever drift apart. Sampling is typed, because an untyped sampler
  passes `sanitize()` and then dies in `evaluate()` — which would quietly bias
  the null by deleting its malformed draws and keeping the survivors. Draws are
  matched to the Sonnet set's call-count and depth distribution so the null is a
  search of the same size and shape. About a quarter of raw draws evaluate to a
  constant cross-section and are replaced, using the feature panel only, never
  forward returns. Each drawn set is written to
  `data/baseline_comparison/random_grammar_seed{N}.txt` so the null is
  inspectable and can be re-scored through `sp500_score_mined_factors.py` like
  any other candidate file.
- **Alpha101** (`configs/alpha101_us.txt`) transcribes Kakushadze
  ([arXiv 1601.00991](https://arxiv.org/abs/1601.00991)). 50 of 101 are
  expressible here. Of the 51 dropped, 49 need `vwap`, `cap` or an industry
  classification the panel does not carry; the remaining two are alpha029, which
  nests 12 deep against the sanitizer's cap of 10, and alpha060, whose two
  `scale()` calls normalise over different cross-sections and so cannot be
  factored out. Every drop is listed in place with its reason.

  The file is **checked against the paper mechanically**, not from memory: all
  101 published formulas were parsed and compared, confirming that no transcribed
  alpha needs a field the panel lacks, that every drop-for-field claim matches
  the published formula, and that all numeric constants match exactly. Operator
  mappings and the deliberate deviations — epsilon-guarded division, `TS_ARGMAX`
  orientation, rounded non-integer windows, and the `adv{d}` reading below — are
  documented in the file header.

  One deviation is an interpretation rather than a transcription. The paper
  defines `adv{d}` as average daily *dollar* volume and `volume` as shares, but
  alphas 7, 17, 21, 39 and 43 compare or divide one against the other. Taken
  literally alpha007 evaluates to -1 on 99% of observations, so those five use
  average daily *share* volume.

## How many survive a factor-zoo correction

The [baseline table](#does-the-llm-add-anything) shows that a best-of-N
t-statistic means nothing without knowing N. This is the next question, asked
one factor at a time: pool everything this repo has scored, and ask which
individual factors clear a hurdle that knows how many were tried.

The pool is 397 scorable factors on a common 3,843-day sample (2001-04 to
2017-12; the missing first year is rolling-window warm-up, shared by every
factor so the maximum is a maximum over one sample):

| group | sets | factors |
|---|---|---:|
| model | Fable 5, Sonnet 5, fundamental | 85 |
| random | five seeded random-grammar draws | 262 |
| published | Alpha101 | 50 |

```bash
python scripts/sp500_build_factor_pool.py   # once, ~66 min, caches every IC series
python scripts/sp500_factor_zoo_hurdle.py
```

### The first surprise: the HAC correction barely matters

Every t-statistic below is a Newey-West t (Bartlett kernel, 21 lags) rather than
the iid t the scoring CLI prints, because daily ICs are not independent draws.
The expectation going in was that this alone would deflate the repo's headline
figures. It does not:

- median |iid t| / |NW t| across the pool: **1.03**
- pool maximum: **10.88** iid against **10.27** Newey-West

Daily cross-sectional rank ICs turn out to be close to serially uncorrelated on
this panel. The inflation in these t-statistics is multiplicity, not
autocorrelation, and saying so is more useful than the tidier story.

### The hurdles

| hurdle | critical value | what it controls |
|---|---:|---|
| random-grammar max \|t\| | 3.51 | the 95th percentile of the bootstrapped maximum \|t\| over the 262 random factors, under a zero-mean null with the same block structure |
| Harvey, Liu and Zhu (2016) | 3.00 | the flat hurdle the cross-sectional literature adopted |
| Benjamini-Hochberg, 5% | p <= 0.033 | false discovery rate across the pool |
| Romano-Wolf stepdown, 5% | 3.65 falling to 3.54 | family-wise error, bootstrapped so the pool's cross-correlation counts |

### Survivors

| group | n | random-max | HLZ | BH | Romano-Wolf |
|---|---:|---:|---:|---:|---:|
| Fable 5 | 20 | 6 | 10 | 12 | 6 |
| Sonnet 5 | 53 | 12 | 19 | 25 | 12 |
| fundamental | 12 | 6 | 7 | 8 | 6 |
| **all model** | **85** | **24** | **36** | **45** | **24** |
| **random grammar** | **262** | **109** | **123** | **186** | **107** |
| **Alpha101** | **50** | **28** | **29** | **38** | **27** |

Random expressions clear every hurdle at a HIGHER rate than the model's factors:
41% against 28% under Romano-Wolf. The published set clears at 54%.

That is not a bug in the correction. It is what a correction cannot do: these
random expressions are not noise. They are functions of price and volume, so a
great many of them are restatements of size, liquidity or short-term reversal,
and those have genuine non-zero mean ICs in sample. A multiple-testing
correction asks "is this mean distinguishable from zero", not "is this an
exploitable edge".

### What the survivors turn out to be

158 factors clear both the empirical bar and Romano-Wolf. **66 of them (42%)
mention `$dollar_volume`.** Their median holdout IC retention is **-29%**; for
the other 92 it is **+59%**. The correction's survivors are dominated by one
effect, spelled many ways, and that effect reverses out of sample.

### The holdout, with the training sign frozen

| hurdle | group | survivors | sign held | median retention | median NW t out of sample |
|---|---|---:|---:|---:|---:|
| random-max | model | 24 | 20/24 | 55.0% | 1.44 |
| random-max | random | 109 | 64/109 | 26.4% | 1.01 |
| random-max | Alpha101 | 28 | **28/28** | **79.0%** | **2.24** |
| Romano-Wolf | model | 24 | 20/24 | 55.0% | 1.44 |
| Romano-Wolf | random | 107 | 64/107 | 34.5% | 1.02 |
| Romano-Wolf | Alpha101 | 27 | **27/27** | 77.5% | 2.06 |

Alpha101 is the only group whose survivors keep their sign unanimously and
retain most of their in-sample IC. The model's survivors sit between the
published set and the null, closer to the null.

### The fundamental factors are counted 63 times

Compustat fundamentals change on the report date and then sit still. A factor
built on `$roa` takes one new value per firm per quarter, but the daily IC
series scores it on every trading day, so the same observation is counted about
63 times. The mean IC is unaffected; the t-statistic is not, because it divides
by the square root of an n that counts repeats.

Re-scored on one IC date per quarter, which is the coarsest grid on which
consecutive observations carry new fundamental information:

```bash
python scripts/sp500_fundamental_quarterly.py
```

| | daily | quarterly |
|---|---:|---:|
| factors with \|t\| > 2 | **10 of 12** | **0 of 12** |
| median t retention against the iid daily t | | **0.11** |
| square-root-rule reference, 1/sqrt(63) | | 0.13 |

The retention lands on the sqrt rule almost exactly, which is what "the daily
t-statistic is counting the same observation 63 times" predicts. Not one of the
twelve fundamental factors survives being measured on independent observations.

Quarter-end is not a neutral day, so the test is repeated at the first, middle
and last trading day of each quarter: median \|t\| of 0.52, 0.33 and 0.41, with
0, 1 and 0 of twelve above 2. The result is not a calendar artifact.

## Did Alpha101 decay after publication

Kakushadze posted "101 Formulaic Alphas" (arXiv:1601.00991) on 2015-12-31, which
gives a pre/post split that needs no judgement call. The 50 transcribed formulas
are scored on 2000-01 to 2015-12, each one's sign is frozen there, and the same
expressions are scored on 2016-01 to 2025-12. The model's own factors and the
random-grammar draws run through the identical split as controls; neither has a
publication date, so their ratio is ordinary out-of-sample decay with nothing
publication-specific in it.

```bash
python scripts/sp500_alpha101_decay.py
```

| set | n | mean IC pre | mean IC post | post/pre | median ratio | sign held |
|---|---:|---:|---:|---:|---:|---:|
| Alpha101 | 50 | +0.00645 | +0.00502 | **77.8%** | 69.8% | 90% |
| model (control) | 85 | +0.00529 | +0.00330 | 62.4% | 59.2% | 81% |
| random grammar (control) | 260 | +0.00540 | +0.00276 | 51.0% | 48.1% | 65% |

| set | block-bootstrap change in mean IC | 95% CI | p |
|---|---:|---|---:|
| Alpha101 | -0.00143 | [-0.00435, +0.00164] | **0.320** |
| model (control) | -0.00199 | [-0.00388, -0.00003] | 0.042 |
| random grammar (control) | -0.00264 | [-0.00480, -0.00055] | 0.015 |

The paired-t and Wilcoxon figures the script also prints treat the factors
within a set as independent observations. They are not, because each set is full
of restatements of one idea, so those p-values are optimistic. The block
bootstrap resamples dates and is the one to read.

**Alpha101 lost 22% of its mean IC after publication, and that loss is not
statistically distinguishable from zero.** McLean and Pontiff (2016, *Journal of
Finance* 71(1), 5-32) put post-publication decay at 58% across the predictors
they study, of which they attribute roughly 26 points to in-sample overfitting
that any out-of-sample window would expose.

Two readings, and the controls decide between them. If the 22% were
publication-driven, the controls should decay less, because nobody published
them. They decay MORE: 38% for the model's factors and 49% for random
expressions, both significant. So the honest reading is that Alpha101 decayed by
less than ordinary out-of-sample attrition on this universe, and there is no
publication effect visible here at all. That is a weaker result than McLean and
Pontiff's, on one published set, one universe and one execution convention, and
it is what the data says.

## Signal horizon

"Score it at horizon h" has two readings, and they answer different questions:

- **lagged**: IC against the single day from open T+h to open T+h+1. How long
  does the factor stay informative? This curve decays, and the half-life is read
  off it.
- **cumulative**: IC against open T+1 to open T+1+h. How much of the move does
  it capture? This curve usually rises, because the return accumulates faster
  than the edge decays.

They coincide at h = 1, where both reduce to the repo's standard label.

```bash
python scripts/sp500_ic_horizon_curve.py
```

Median |mean IC| on the training window:

| set | convention | h=1 | h=5 | h=10 | h=21 | h=63 |
|---|---|---:|---:|---:|---:|---:|
| Alpha101 | lagged | 0.00607 | 0.00168 | 0.00146 | 0.00129 | 0.00232 |
| Alpha101 | cumulative | 0.00607 | 0.00757 | 0.00742 | 0.00719 | 0.00695 |
| model | lagged | 0.00474 | 0.00289 | 0.00213 | 0.00247 | 0.00409 |
| model | cumulative | 0.00474 | 0.00913 | 0.01064 | 0.01038 | 0.01248 |
| random | lagged | 0.00509 | 0.00457 | 0.00402 | 0.00370 | 0.00474 |
| random | cumulative | 0.00509 | 0.00727 | 0.01026 | 0.01292 | 0.03166 |

Median Newey-West t on the lagged curve tells the same story more sharply:
Alpha101 goes 4.50, 1.57, 1.04, 1.06, 1.93 across the five horizons, while the
random pool barely moves (2.86, 2.83, 2.62, 2.47, 2.99).

That contrast is the finding. Alpha101's edge is genuinely short-horizon and
genuinely decays: 42 of 50 halve inside the 63-day grid, with a median half-life
of **4.0 days**, and its cumulative curve is flat past a week, so there is
nothing to be gained by holding longer. The random pool does not decay at all,
because 185 of 270 of its members are not predicting anything time-varying, they
are slow-moving characteristics: their cumulative IC keeps climbing to 0.032 at
63 days, which is what a size or liquidity exposure looks like measured this
way. The model's factors sit between the two: 43 of 85 never halve, median
half-life 4.4 days for those that do.

This half-life is what the evolutionary loop's archive uses for its horizon
axis. On the pool as a whole the buckets come out lopsided (77 fast, 15 medium,
177 slow among hurdle survivors), which is worth knowing before reading a niche
count as evidence of diversity.

## Is the model paraphrasing

The [memorization test](#the-memorization-test) asks this in signal space: how
correlated is each mined factor with its closest published alpha? That measure
misses a restatement whose numbers happen to differ, and it flags coincidences,
because everything in equities co-moves. `quantaalpha_us/factors/ast_tools.py`
adds an independent axis that never touches the data: canonicalise both
expressions into trees, and measure the largest complete sub-expression they
share and the Zhang-Shasha edit distance between them.

Reproduce in ten seconds, with no WRDS entitlement:

```bash
python scripts/sp500_structure_report.py
```

Neither statistic means anything alone, because a longer expression shares more
with everything by accident. The comparison is against random-grammar draws,
which never saw the literature, so whatever they score is the level of
structural overlap that costs nothing to explain.

| statistic (median) | model sets | random grammar | permutation p |
|---|---:|---:|---:|
| largest shared subtree with any Alpha101, nodes | 1.00 | 1.00 | 1.000 |
| that subtree as a share of the candidate's own tree | 0.20 | 0.10 | < 0.0001 |
| edit distance to the nearest Alpha101 formula | 8.00 | 10.00 | 0.017 |
| normalized edit distance | 0.46 | 0.55 | < 0.0001 |
| tree size, nodes | 8.00 | 12.00 | < 0.0001 |

The model's factors sit structurally closer to the published set on every
measure. Most of that is size: the model writes shorter expressions, and a
shorter expression is closer to everything. Pairing each model factor with the
random draws within two nodes of its own size removes almost all of it:

- median excess shared subtree against size-matched random: **+0.00 nodes**
- median excess edit distance against size-matched random: **-1.00 edits**
- model factors sharing a strictly larger subtree than their size-matched
  random median: **30 of 86**

So the typical mined factor is not a paraphrase. The tail is a different matter,
and the tail is what a paraphrase check is for:

| shared subtree of at least | model sets | random grammar |
|---|---:|---:|
| 3 nodes | 30% | 12% |
| 4 nodes | 7% | 0% |
| 5 nodes | 6% | 0% |
| 6 nodes | 2% | 0% |
| 7 nodes | 2% | 0% |

Above four nodes the random baseline is empty and the model is not. The largest
overlap found is seven nodes: Sonnet 5's
`CS_RANK((($close - $low) - ($high - $close)) / ($high - $low + 1e-8))` shares
the whole close-location-value numerator with a transcribed Alpha101 formula,
six edits away from it. That is a restatement of a published construct, offered
without attribution.

This is why the evolutionary loop gates on structure and not only on
correlation: a candidate sharing a complete sub-expression of five or more nodes
with any published alpha is rejected before it is scored. That threshold rejects
6% of the existing model-mined factors and 0% of the random ones, so it is a
real constraint rather than decoration.

## Universe

Scoring is restricted to **point-in-time S&P 500 membership**, joined on
`(date, permno)` before any factor is computed.

This is not cosmetic. `daily_bars.parquet` carries every name that was ever a
constituent over 2000-2025, which is ~745 names per date against the index's
~502. On 2005-06-15, 315 of the 814 names in the panel (39%) were not in the
S&P 500 that day, and they are future members as often as former ones. Ranking
a factor cross-sectionally over that panel scores it on a universe no strategy
could have held, and it inflates significance: the best candidate measured
|t| = 8.21 unfiltered against 7.56 filtered.

`--membership none` scores the raw panel and says so loudly; a missing
membership file is a hard error rather than a silent skip.

## Data source

Market data comes from **CRSP via WRDS**. Every call site this repo actually
uses — daily bars, bulk daily bars, and the ticker mapping — is served by
`quantaalpha_us/data/crsp_client.py`, which also carries PERMNO identity,
delisting returns and point-in-time membership that a plain vendor EOD feed
does not.

`build_market_data_client()` returns that client and raises if
`CRSP_USERNAME`/`CRSP_API_KEY` are absent, rather than silently degrading to a
weaker source. An EODHD fallback was removed in 2026-09: it was never the source
of any published result here, and a second vendor path that nothing exercises is
a maintenance and credential liability rather than resilience.

Running this repo therefore requires a WRDS entitlement.

CRSP data is licensed and is deliberately not committed: the repo ships code,
derived factor scores and figures, not raw vendor data.

### Which CRSP tape

`crsp_client._daily_table()` resolves the daily stock file to **`crsp.dsf_v2`**
, CRSP's Flat File Format 2.0, the "CIZ" tape , and falls back to the legacy
`crsp.dsf` only when `dsf_v2` is absent from the entitlement. Berkeley's WRDS
subscription carries `dsf_v2`, so every number in this repo is on CIZ.

This matters more than a table name usually would. CRSP shipped the last release
of the legacy SIZ tape in January 2025 and now updates only CIZ, and the two are
not the same history. Schwarz, Walter and Weiss, *Rewriting CRSP's History:
Impact of Altered Monthly Returns on Asset Pricing*, Journal of Financial and
Quantitative Analysis (2026), measure the switch: **9.62%** of monthly returns
change by more than 1 bp, mostly because payouts now reinvest on the ex-date
rather than at month end, and **11.43%** of monthly long-short returns move by
more than 10 bp, concentrated in early periods, NBER recessions and
return-based sorts. Their headline finding is reassuring for a study like this
one (average premia and their significance survive the switch), but the
per-portfolio differences are large enough that a result reproduced on the other
tape will not match to the last basis point.

Two consequences worth stating plainly:

- Anyone reproducing these ICs on `crsp.dsf` should expect small differences,
  and they are the tape, not the code.
- The sibling trend-following repo (`VOO_Backtest`) is built on the **legacy SIZ
  tape**, with CIZ wired in only as a comparison path (`crsp_v2.py`). Numbers
  are therefore not directly comparable between the two repos, and neither is
  wrong.

## Known limits

- Factor quality, not tooling, is the binding limit: mean ICs of 0.003-0.011 are weak against 0.02-0.05 for a decent published factor. The large t-statistics are worse than inflated by sample length — they are **below the noise floor of a search this size**. Random expressions drawn from the same grammar reach best |t| of 9.11 to 12.41 on this panel, so a |t| of 8 is not evidence of a factor at all; see [Does the LLM add anything?](#does-the-llm-add-anything).
- Research quality still depends on the quality and timeliness of external data providers
- Daily signals and retail-oriented execution assumptions are intentionally conservative and do not represent intraday HFT infrastructure
- LLM factor generation is bounded and audited, but it still needs human judgment before production use

## Validation notes (2026-08 revision)

- Gate-1 now uses the actual Deflated Sharpe Ratio (Bailey and Lopez de Prado 2014): the probability that the strategy's true Sharpe exceeds the expected maximum Sharpe of `n_trials` zero-skill strategies, adjusted for skewness and kurtosis. The threshold is a probability (default 0.95). The previous implementation was a heuristic penalty and overstated significance.
- Gate-6 (factor stability) is now measured: per-fold Spearman rank ICs of each baseline factor against realized entry-to-exit returns, top-3 factor sets per fold, mean Jaccard overlap across consecutive folds. It was previously hard-coded to pass.
- Holdings that drop out of the tradable context (index exit, missing bar) are now explicitly liquidated at the entry open with transaction costs; they previously converted to cash silently with no cost or turnover.

## Notes

The repository is maintained at `Aroesler1/LLMStrat` and keeps the `quantaalpha_us` module path for runtime compatibility with earlier internal tooling

## Project Scope

This codebase covers the full lifecycle of a daily US large-cap systematic strategy:

- historical universe construction
- daily OHLCV ingestion and quality control
- baseline portfolio construction
- walk-forward backtesting
- LLM-assisted factor ideation
- paper and live trading orchestration
- pre-trade and post-trade risk checks

It is not a high-frequency system and does not attempt to model intraday microstructure beyond what is realistic for a daily rebalance process.

## Architecture

### Data

- `quantaalpha_us/data`
  - CRSP/WRDS client
  - membership builders
  - data quality checks

### Research

- `quantaalpha_us/pipeline`
  - baseline feature generation
  - signal generation
- `quantaalpha_us/backtest`
  - point-in-time universe handling
  - walk-forward runner
  - transaction cost model
  - validation gates

### LLM Runtime

- `quantaalpha_us/llm`
  - request budgeting
  - fallback handling
  - factor extraction
  - expression sanitization

### Trading

- `quantaalpha_us/trading`
  - Alpaca REST adapter
  - operational risk controls
  - post-trade reconciliation checks

### Entry Points

- `scripts/`
  - membership build
  - backfill and daily ingest
  - research
  - factor mining
  - reporting
  - orchestration
  - trade submission

### Validation

- `tests/`
  - unit coverage over the current stack

## Technical Implementation

The system is implemented as a Python-based research and execution stack with explicit separation between data, research, LLM runtime, and trading concerns.

Key technical characteristics:

- configuration-driven workflows through YAML
- parquet and CSV artifacts for reproducible intermediate datasets
- point-in-time universe handling rather than static ticker lists
- explicit CLI entrypoints for each stage of the pipeline
- deterministic walk-forward evaluation instead of a single in-sample backtest
- automated checks around data quality, turnover, concentration, and research promotion
- broker integration through a REST execution layer rather than notebook-driven manual trading

The code is structured so that research artifacts, signals, and trade actions can be produced from the same underlying pipeline rather than from disconnected scripts.

## Design Priorities

Several design choices define the project:

- data realism before model complexity
- explicit point-in-time universe handling
- research outputs that can fail promotion rather than silently pass
- operational controls around factor mining instead of unconstrained prompting
- retail-aware execution assumptions instead of idealized frictionless backtests

The result is a system that aims to be honest about what is known, what is approximated, and what still needs empirical validation.

## Tools And Technologies

Beyond LLM usage, this project uses and demonstrates familiarity with:

- Python for the full research and execution stack
- `pandas` for feature engineering, panel manipulation, and research outputs
- parquet-based data artifacts for reproducible local research datasets
- CRSP through WRDS for research-grade historical membership and price data
- Alpaca REST APIs for paper and live execution
- YAML-based configuration for research, mining, and trading profiles
- CLI-oriented orchestration through standalone Python entrypoints
- `pytest` for automated test coverage
- Git and GitHub for versioned development and deployment of research code

From a systems perspective, the project required work across:

- external data integration
- schema normalization
- data quality validation
- portfolio construction logic
- transaction cost modeling
- broker API integration
- runtime fault handling
- test-driven refactoring

## Data Modes

The project supports two operating modes.

### Research-Grade Mode

This mode uses a proper historical membership history sourced from CRSP
through WRDS.

This is the intended mode for serious walk-forward research.

### Approximate Mode

This mode builds a constant-membership S&P 500 approximation from a current constituent snapshot and is retained for:

- bring-up
- pipeline validation
- engineering checks
- low-cost research scaffolding

It is useful operationally, but it is not treated as a substitute for true historical membership data.

## Current Capabilities

At the current stage, the repo supports:

- CRSP market data via WRDS
- historical or approximate S&P 500 membership construction
- daily bar backfill and coverage reporting
- baseline signal generation
- walk-forward research with validation gates
- LLM factor mining with exact model enforcement
- retail-oriented execution simulation
- paper and live trading entrypoints through Alpaca

The main open problem is not infrastructure completeness. The remaining challenge is improving strategy quality enough to satisfy the research gates under realistic assumptions.

## Research Philosophy

The research loop is deliberately conservative.

The backtest includes:

- next-day open execution alignment
- retained cash buffer
- minimum trade-size filtering
- fractional-share handling
- participation limits relative to ADV
- liquidity-aware transaction cost modeling
- sector caps in portfolio construction

The goal is not to create a perfect live execution simulator. The goal is to avoid the far more common failure mode of producing a backtest that is materially cleaner than a retail-accessible implementation could ever achieve.

## LLM Factor Mining

The factor-mining subsystem is treated as a constrained research tool, not a source of unchecked strategy logic.

Current runtime behavior:

- exact primary model: `gpt-5.4`
- exact fallback model: `gemini-3.1-pro`
- OpenAI-compatible endpoint requirement
- strict model-catalog enforcement
- JSON-only output expectation
- sanitizer pass before any expression is accepted
- request and token budgets
- early halt on low-quality output

This is intended to keep model experimentation productive without allowing generated expressions to quietly degrade research quality.

## Setup

```bash
cd /absolute/path/to/QuantaAlpha_US
./bootstrap.sh
source .venv/bin/activate
cp .env.example .env
set -a && source .env && set +a
```

Typical environment configuration includes:

- `CRSP_USERNAME`
- `CRSP_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `ALPACA_PAPER_API_KEY`
- `ALPACA_PAPER_API_SECRET`
- `ALPACA_LIVE_API_KEY`
- `ALPACA_LIVE_API_SECRET`

## Core Artifacts

The central pipeline artifacts are:

- `data/us_equities/reference/sp500_membership_daily.parquet`
- `data/us_equities/reference/gics_sectors.csv`
- `data/us_equities/reference/ticker_mapping.csv`
- `data/us_equities/processed/daily_bars.parquet`

## External Interfaces

The project interacts with several real external systems:

- WRDS / CRSP for research-grade historical data
- OpenAI-compatible chat-completions endpoints for factor mining
- Alpaca for execution and account state

That matters because the repository is not only a modeling exercise. It includes API integration, operational failure handling, and the practical engineering needed to move from research outputs to broker-facing actions.

## Representative Workflow

### 1. Build Membership

```bash
python scripts/sp500_build_membership.py
```

Source selection can also be made explicit:

```bash
python scripts/sp500_build_membership.py --source crsp
python scripts/sp500_build_membership_approx.py
```

### 2. Backfill Daily Bars

```bash
python scripts/sp500_backfill_history.py
python scripts/sp500_data_coverage_report.py
```

### 3. Generate Signals

```bash
python scripts/sp500_generate_signals.py --config configs/backtest_sp500_research.yaml
```

### 4. Run Walk-Forward Research

```bash
python scripts/sp500_run_research.py --config configs/backtest_sp500_research.yaml
```

### 5. Run LLM Factor Mining

```bash
python scripts/sp500_run_factor_mining.py --config configs/llm_sp500.yaml --live-call
```

### 6. Submit a Paper Rebalance

```bash
python scripts/trade_once.py --config configs/paper_sp500.yaml --dry-run
python scripts/trade_once.py --config configs/paper_sp500.yaml
```

### 7. Orchestrate Scheduled Modes

```bash
python scripts/sp500_orchestrator.py --mode research --run-date 2026-03-10
python scripts/sp500_orchestrator.py --mode paper --run-date 2026-03-10 --dry-run
python scripts/sp500_orchestrator.py --mode live --run-date 2026-03-10 --dry-run
```

## Script-Level Flow

The main CLI entrypoints correspond to specific parts of the pipeline:

- `scripts/sp500_build_membership.py`
  - builds historical membership from CRSP
- `scripts/sp500_build_membership_approx.py`
  - builds a constant-membership approximation from a current snapshot
- `scripts/sp500_backfill_history.py`
  - backfills historical daily bars
- `scripts/sp500_data_coverage_report.py`
  - measures member-day coverage of the dataset
- `scripts/sp500_ingest_daily.py`
  - performs daily incremental updates
- `scripts/sp500_generate_signals.py`
  - converts market data into target portfolio weights
- `scripts/sp500_run_research.py`
  - runs walk-forward research and validation
- `scripts/sp500_run_factor_mining.py`
  - generates and filters candidate factor expressions
- `scripts/trade_once.py`
  - submits one rebalance cycle
- `scripts/sp500_orchestrator.py`
  - coordinates multi-step scheduled runs

That division is deliberate. Each script has a narrow responsibility, which keeps the project easier to test, inspect, and operate.

## Research Outputs

A research run writes artifacts under `data/results/research/...`, including:

- walk-forward returns
- fold metadata
- research summary
- validation gate results

The validation layer is a first-class part of the project. A run completing successfully is not treated as equivalent to a strategy passing research review.

## Operational Tooling

Additional operational scripts include:

Daily report:

```bash
python scripts/sp500_daily_report.py
```

Kill switch:

```bash
python scripts/kill_switch.py --config configs/paper_sp500.yaml --level 2 --reason "manual test" --yes
```

## Why This Project Exists

The point of this repository is not to present a polished research result. It is to show a full-stack, research-to-execution implementation that takes data integrity, execution realism, and model governance seriously.

In practical terms, that means:

- a real historical universe path exists
- a lower-fidelity approximation path exists when needed
- research and trading share the same operational assumptions where possible
- LLM usage is constrained and reviewable
- promotion depends on validation, not narrative

## Tests

```bash
pytest
```

## Summary

`QuantaAlpha_US` is best understood as a serious systematic trading workbench rather than a toy backtest or a generic LLM wrapper. It combines data engineering, research discipline, execution awareness, and model-safety controls in a single repo.

The remaining work is primarily on strategy quality, not scaffolding. That is the right stage for a project of this kind to be in.
