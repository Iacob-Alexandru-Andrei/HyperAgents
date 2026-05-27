# HyperAgents Paper Review TOON Experiment Summary

## Executive Summary

This checkpoint consolidates the completed 30-sample `paper_review` TOON runs in
`/home/ubuntu/git-repos/HyperAgents-toon`.

The loader-only CSV -> TOON experiment validated the pipeline, but it did not
reduce prompt tokens because the LLM still receives the same raw `paper_text`.
The useful token savings came only from prompt-side compression.

The best completed prompt-side tradeoff is `retrieval_toon`: it used 258,237
total tokens, a 67.50% reduction from the raw CSV baseline, while scoring
22/30 accuracy. `toon_prompt_neutral` scored higher at 23/30, but used 295,174
tokens and appears more prompt-sensitive.

## Completed Run Comparison

Metrics below were recomputed from each run's `report.json` and
`token_usage.jsonl`. Every completed run has 30 token records for 30 samples.

| run | description | accuracy | prompt tokens | completion tokens | total tokens | total delta vs raw | total delta pct | accept P/R | reject P/R |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `csv_baseline` | Raw CSV baseline | 18/30 = 0.600 | 729,125 | 65,458 | 794,583 | 0 | 0.00% | 1.000 / 0.200 | 0.556 / 1.000 |
| `toon_loader` | TOON loader parity | 20/30 = 0.667 | 729,125 | 58,808 | 787,933 | -6,650 | -0.84% | 1.000 / 0.333 | 0.600 / 1.000 |
| `toon_prompt_compact` | Compact prompt TOON | 15/30 = 0.500 | 168,446 | 40,420 | 208,866 | -585,717 | -73.71% | 0.000 / 0.000 | 0.500 / 1.000 |
| `toon_prompt_richer` | Rich prompt TOON | 15/30 = 0.500 | 232,962 | 57,743 | 290,705 | -503,878 | -63.41% | 0.000 / 0.000 | 0.500 / 1.000 |
| `toon_prompt_neutral` | Neutral rich prompt TOON | 23/30 = 0.767 | 232,302 | 62,872 | 295,174 | -499,409 | -62.85% | 1.000 / 0.600 | 0.700 / 0.933 |
| `core_toon_sidecar` | Raw core text + TOON sidecar | 15/30 = 0.500 | 262,699 | 60,041 | 322,740 | -471,843 | -59.38% | 0.000 / 0.000 | 0.500 / 1.000 |
| `retrieval_toon` | Retrieval TOON | 22/30 = 0.733 | 201,490 | 56,747 | 258,237 | -536,346 | -67.50% | 0.733 / 0.733 | 0.733 / 0.733 |

## Interpretation

`toon_loader` is a loader integration result, not a compression result. Prompt
tokens are exactly equal to `csv_baseline` at 729,125 because the CSV row is
decoded before prompting and the same markdown paper text is sent to the model.
The small total-token delta comes from nondeterministic completion length, not
TOON input compression.

`toon_prompt_compact`, `toon_prompt_richer`, and `core_toon_sidecar` prove that
prompt-side compression can cut token use substantially, but the paper evidence
selection and reviewer prompt matter. These runs collapsed into all-reject or
mostly-reject behavior despite large token savings.

`toon_prompt_neutral` is the highest-accuracy completed prompt-side run, but it
depends on neutralizing the original harsh reviewer prompt. It is useful
evidence that prompt framing affects this benchmark, but it is not the cleanest
format comparison.

`retrieval_toon` is the best completed result for the paper story: it keeps a
balanced accept/reject distribution, preserves both accept and reject precision
and recall at 0.733, and still cuts total tokens by 67.50%.

## Incomplete Run

`two_stage_toon` is not a completed benchmark result. It has:

- 11 token records in `runs/two_stage_toon/token_usage.jsonl`
- 6 partial chat histories under `runs/two_stage_toon/agent_evals/`
- no `runs/two_stage_toon/predictions.csv`
- no `runs/two_stage_toon/report.json`

It should stay excluded from aggregate comparisons until rerun from scratch or
resumed with a clearly documented method. Because it makes two model calls per
paper, it must beat `retrieval_toon` by enough accuracy to justify the extra
latency and token accounting complexity.

## Recommended Next Experiment

Do not spend more NVIDIA inference tokens on two-stage TOON first. The next
useful experiment is to tune `retrieval_toon` in one-call form:

- keep the retrieval-style packet structure
- adjust chunk count and chunk character limit
- keep the neutral reviewer framing fixed
- compare against `retrieval_toon` and `toon_prompt_neutral`

That path directly targets the current frontier: preserve the 60% to 70% token
savings while recovering the last one or two accuracy points.
