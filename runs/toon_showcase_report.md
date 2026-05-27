# TOON Showcase Report

## Headline Result

On the fixed 30-sample HyperAgents `paper_review` eval set, retrieval TOON
matched the raw neutral reviewer at 24/30 correct predictions while reducing
total LLM tokens by 66.84%.

This is the controlled comparison needed for the TOON claim: the model, dataset,
reviewer framing, temperature, and worker count were held fixed. The only
intentional difference was the paper representation sent to the model.

## Controlled Runs

| run | paper representation | reviewer framing | accuracy | prompt tokens | completion tokens | total tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `raw_neutral_baseline` | raw `paper_text` | `reviewer_system_prompt_base` | 24/30 = 0.800 | 728,585 | 69,635 | 798,220 |
| `retrieval_toon_controlled` | retrieval TOON packet | `reviewer_system_prompt_base` | 24/30 = 0.800 | 201,490 | 63,186 | 264,676 |

## Deltas

| metric | raw_neutral | retrieval_toon | delta |
| --- | ---: | ---: | ---: |
| correct predictions | 24 | 24 | 0 |
| overall accuracy | 0.8000 | 0.8000 | 0.0000 |
| prompt tokens | 728,585 | 201,490 | -527,095 (-72.35%) |
| completion tokens | 69,635 | 63,186 | -6,449 (-9.26%) |
| total tokens | 798,220 | 264,676 | -533,544 (-66.84%) |

## Label Behavior

| label metric | raw_neutral | retrieval_toon | delta |
| --- | ---: | ---: | ---: |
| accept precision | 0.7368 | 0.8462 | +0.1093 |
| accept recall | 0.9333 | 0.7333 | -0.2000 |
| reject precision | 0.9091 | 0.7647 | -0.1444 |
| reject recall | 0.6667 | 0.8667 | +0.2000 |

Both runs predicted both labels and both labels had nonzero recall. Retrieval
TOON traded some accept recall for reject recall, but the total success rate was
identical to the raw neutral baseline.

## Success Criteria Audit

| criterion | result | status |
| --- | --- | --- |
| TOON total tokens at least 50% lower than raw | 66.84% lower | pass |
| TOON accuracy no worse than raw by more than 1 sample | equal, 24/30 vs 24/30 | pass |
| TOON predicts both `Accept` and `Reject` | prediction distribution is 43.3% accept, 56.7% reject | pass |
| Accept and reject recall are nonzero | accept recall 0.7333, reject recall 0.8667 | pass |
| Each included run has 30 predictions | both runs have 30 prediction rows | pass |
| Each included run has 30 token records | both runs have 30 token records | pass |
| Each included run has `report.json` | both reports generated | pass |

## Files

- Raw baseline run: `runs/raw_neutral_baseline/`
- Retrieval TOON run: `runs/retrieval_toon_controlled/`
- Pairwise comparison: `runs/raw_neutral_vs_retrieval_toon.md`
- Experiment plan: `runs/toon_showcase_experiment_plan.md`
- Prior result summary: `runs/toon_experiment_summary.md`

## Interpretation

Loader-level TOON still matters as an integration result, but it does not save
prompt tokens in `paper_review` because the raw markdown paper is what reaches
the LLM. The token savings come from prompt-side TOON: a retrieval-style packet
retains title, abstract, ranked evidence chunks, and key tables while dropping
back matter and lower-value sections.

The controlled result supports the intended claim for the 30-sample eval set:
retrieval TOON can preserve raw-input success rate while using about one third
of the total tokens.

## Caveats

- This is a 30-sample showcase, not a high-powered statistical study.
- Nemotron can still show run-to-run variance even with `temperature=0`.
- A larger 60- or 100-sample confirmation run would strengthen the claim before
  using it as a final paper result.
- `two_stage_toon` remains incomplete and is excluded from this report.
