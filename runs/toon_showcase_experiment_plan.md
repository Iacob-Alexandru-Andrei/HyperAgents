# TOON Showcase Experiment Plan

## Goal

Demonstrate that a TOON prompt representation can materially reduce LLM token
usage while matching the success rate of the normal raw-paper input on the
HyperAgents `paper_review` benchmark.

The claim must be based on a fair controlled comparison:

- same model
- same 30-sample deterministic eval set
- same reviewer framing
- same temperature and worker settings
- only the paper representation changes

Existing results show that loader-only TOON has no prompt-token effect, and that
prompt-side TOON can save tokens. The remaining work is to make the comparison
clean enough to support a paper/review claim.

## Success Criteria

Primary success criteria for the 30-sample showcase:

- Token savings: TOON total tokens are at least 50% lower than the matched raw
  baseline.
- Success rate: TOON accuracy is no worse than the matched raw baseline by more
  than 1 sample out of 30.
- Label behavior: TOON predicts both `Accept` and `Reject`; neither accept
  recall nor reject recall may be zero.
- Run quality: each completed run has 30 predictions, 30 token records, and a
  `report.json`.

Recommended stronger confirmation, if token budget allows:

- Repeat the winning pair on 60 or 100 deterministic samples.
- For 60 samples, allow at most 2 fewer correct predictions than raw.
- For 100 samples, allow at most 3 fewer correct predictions than raw.

## Current Evidence

Completed current-state runs are summarized in `runs/toon_experiment_summary.md`.

Key observations:

- `toon_loader` has exactly the same prompt tokens as `csv_baseline`
  (`729,125`) because it decodes to the same raw `paper_text` before prompting.
- `retrieval_toon` is the best completed prompt-side TOON tradeoff so far:
  `22/30` accuracy, `258,237` total tokens, and 67.50% total-token reduction
  versus `csv_baseline`.
- `toon_prompt_neutral` is the highest-accuracy prompt-side run so far:
  `23/30`, but it changes reviewer framing and uses more tokens than
  `retrieval_toon`.
- `two_stage_toon` is incomplete and excluded from claims.

The current `csv_baseline` vs `retrieval_toon` comparison is not sufficient for
the final claim because `csv_baseline` uses the harsher
`reviewer_system_prompt_neg`, while `retrieval_toon` uses
`reviewer_system_prompt_base`.

## Experiment Matrix

Run these in order. Stop early only if a decision gate explicitly says to stop.

| order | run_id | input representation | reviewer framing | purpose |
| ---: | --- | --- | --- | --- |
| 1 | `raw_neutral_baseline` | raw `paper_text` | `reviewer_system_prompt_base` | fair normal-input baseline |
| 2 | `retrieval_toon_controlled` | retrieval TOON packet | `reviewer_system_prompt_base` | main TOON candidate |
| 3 | `retrieval_toon_more_chunks` | retrieval TOON, more chunks | `reviewer_system_prompt_base` | recover missed evidence |
| 4 | `retrieval_toon_longer_chunks` | retrieval TOON, longer chunks | `reviewer_system_prompt_base` | preserve local context |
| 5 | `retrieval_toon_tables_plus_results` | retrieval TOON, table/result emphasis | `reviewer_system_prompt_base` | strengthen result evidence |
| 6 | `raw_neutral_confirm` | raw `paper_text` | `reviewer_system_prompt_base` | confirmation pair, if needed |
| 7 | `retrieval_toon_winner_confirm` | best TOON variant | `reviewer_system_prompt_base` | confirmation pair, if needed |

## Implementation Prep

Add one raw neutral control agent:

- Create `baselines/ai_reviewer_raw_neutral/agent.py`.
- Copy `baselines/ai_reviewer/agent.py`.
- Change only `reviewer_system_prompt = reviewer_system_prompt_neg` to
  `reviewer_system_prompt = reviewer_system_prompt_base`.
- Keep the raw paper prompt and JSON extraction behavior identical.

Add up to three retrieval variants only after the controlled run finishes:

- `baselines/ai_reviewer_retrieval_toon_more_chunks/agent.py`
  calls `build_retrieval_packet_toon(..., max_chunks=12, chunk_char_limit=2800)`.
- `baselines/ai_reviewer_retrieval_toon_longer_chunks/agent.py`
  calls `build_retrieval_packet_toon(..., max_chunks=8, chunk_char_limit=4200)`.
- `baselines/ai_reviewer_retrieval_toon_tables_plus_results/agent.py`
  uses the same call shape as `retrieval_toon`, but after adding result/table
  keyword weighting in `domains/paper_review/prompt_packet.py`.

Keep all variants one-call agents. Do not resume `two_stage_toon` until the
one-call frontier is exhausted.

## Commands

Use the same dependency wrapper already proven by the existing TOON tests and
runs:

```bash
export OPENAI_API_KEY="$NVIDIA_API_KEY"
export UV_CACHE_DIR=/tmp/hyperagents-toon-uv-cache
export RUNPY="uv run --with litellm==1.74.9 --with python-dotenv --with requests==2.32.4 --with backoff==2.2.1 --with pandas==2.3.2 python"
```

Run the fair raw baseline:

```bash
$RUNPY -m domains.harness \
  --domain paper_review \
  --agent_path baselines/ai_reviewer_raw_neutral/agent.py \
  --dataset_path domains/paper_review/dataset_eval.csv \
  --num_samples 30 \
  --num_workers 1 \
  --output_dir runs \
  --run_id raw_neutral_baseline \
  --token_log runs/raw_neutral_baseline/token_usage.jsonl

python3 -m domains.report \
  --domain paper_review \
  --dname runs/raw_neutral_baseline
```

Run the controlled TOON candidate:

```bash
$RUNPY -m domains.harness \
  --domain paper_review \
  --agent_path baselines/ai_reviewer_retrieval_toon/agent.py \
  --dataset_path domains/paper_review/dataset_eval.csv \
  --num_samples 30 \
  --num_workers 1 \
  --output_dir runs \
  --run_id retrieval_toon_controlled \
  --token_log runs/retrieval_toon_controlled/token_usage.jsonl

python3 -m domains.report \
  --domain paper_review \
  --dname runs/retrieval_toon_controlled
```

Compare the controlled pair:

```bash
python3 scripts/compare_toon_runs.py \
  --csv runs/raw_neutral_baseline \
  --toon runs/retrieval_toon_controlled \
  --out runs/raw_neutral_vs_retrieval_toon.md \
  --title "Raw Neutral vs Retrieval TOON" \
  --csv-label raw_neutral \
  --toon-label retrieval_toon
```

Only run tuning variants if `retrieval_toon_controlled` misses the success-rate
criterion or has obvious label imbalance.

Variant commands follow the same shape:

```bash
$RUNPY -m domains.harness \
  --domain paper_review \
  --agent_path baselines/ai_reviewer_retrieval_toon_more_chunks/agent.py \
  --dataset_path domains/paper_review/dataset_eval.csv \
  --num_samples 30 \
  --num_workers 1 \
  --output_dir runs \
  --run_id retrieval_toon_more_chunks \
  --token_log runs/retrieval_toon_more_chunks/token_usage.jsonl

python3 -m domains.report \
  --domain paper_review \
  --dname runs/retrieval_toon_more_chunks
```

Repeat for `retrieval_toon_longer_chunks` and
`retrieval_toon_tables_plus_results` if needed.

## Decision Gates

Gate 1: after `raw_neutral_baseline`.

- If raw neutral accuracy is lower than 18/30, keep it as the fair baseline but
  report that neutral framing weakens the raw reviewer.
- If raw neutral accuracy is at least 18/30, use it as the main normal-input
  comparator.

Gate 2: after `retrieval_toon_controlled`.

- If TOON is within 1 correct prediction of raw neutral and saves at least 50%
  total tokens, stop tuning and write the showcase report.
- If TOON saves tokens but is worse by 2-4 samples, run the three one-call
  retrieval tuning variants.
- If TOON saves less than 50%, inspect packet size before running more model
  calls.
- If TOON collapses to one label, adjust reviewer wording or evidence selection
  before rerunning.

Gate 3: after tuning variants.

- Select the highest-accuracy TOON variant that saves at least 50% total tokens
  and has nonzero recall for both labels.
- If no variant meets criteria, report the negative result honestly and pivot to
  either a larger eval set or a different domain with more structured prompt
  input.

Gate 4: confirmation.

- If the winning 30-sample result is strong, rerun only the matched raw and TOON
  pair as `raw_neutral_confirm` and `retrieval_toon_winner_confirm`.
- If confirmation preserves the same non-inferiority result, use the confirmed
  pair for the final headline.

## Final Deliverables

Write `runs/toon_showcase_report.md` with:

- controlled run table for raw neutral and each TOON candidate
- accuracy deltas and per-label precision/recall deltas
- prompt/completion/total token deltas
- per-sample token deltas for the winning pair
- clear statement of whether TOON met the non-inferiority criterion
- note that loader-only TOON validates the format but does not save prompt
  tokens in `paper_review`

Suggested headline format if the controlled run succeeds:

> On the fixed 30-sample `paper_review` eval set, retrieval TOON matched the raw
> neutral reviewer within one sample while reducing total LLM tokens by X%.

If the controlled run does not succeed, do not claim equivalent success rate.
Instead report:

> TOON produced X% token savings, but the best controlled prompt-side variant
> trailed raw input by Y samples; more evidence-retention tuning is needed before
> claiming quality parity.

## Verification Checklist

Before making any final claim, verify:

- `python3 -m domains.report` has produced `report.json` for every included run.
- Every included run has exactly 30 rows in `predictions.csv`.
- Every included one-call run has exactly 30 lines in `token_usage.jsonl`.
- The final comparison excludes `two_stage_toon` unless it is rerun to
  completion.
- The final report compares only runs with the same reviewer framing.
- The final report states both token savings and success-rate delta.
