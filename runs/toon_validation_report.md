# TOON Validation Report

## Question

Does the TOON swap preserve paper-review behavior while reducing token usage?

## Runs Checked

| comparison | samples | raw correct | TOON correct | accuracy delta | total token delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `raw_neutral_baseline` vs `retrieval_toon_controlled` | 30 | 24/30 | 24/30 | 0.0000 | -533,544 (-66.84%) |
| `raw_neutral_repeat1` vs `retrieval_toon_repeat1` | 30 | 26/30 | 25/30 | -0.0333 | -538,674 (-67.71%) |
| `raw_neutral_60` vs `retrieval_toon_60` | 60 | 50/60 | 50/60 | 0.0000 | -1,056,686 (-68.21%) |

The prompt-token reduction is stable across the controlled checks: -72.35%, -72.35%, and -72.98%.

## 60-Sample Confirmation

The 60-sample deterministic subset is balanced at 30 accept and 30 reject papers.

| metric | raw neutral | retrieval TOON | delta |
| --- | ---: | ---: | ---: |
| overall accuracy | 0.8333 | 0.8333 | 0.0000 |
| accept precision | 0.7632 | 0.7778 | +0.0146 |
| accept recall | 0.9667 | 0.9333 | -0.0333 |
| reject precision | 0.9545 | 0.9167 | -0.0379 |
| reject recall | 0.7000 | 0.7333 | +0.0333 |

## Flip Analysis

On the 60-sample confirmation run:

| category | count |
| --- | ---: |
| raw correct, TOON correct | 44 |
| raw correct, TOON wrong | 6 |
| raw wrong, TOON correct | 6 |
| raw wrong, TOON wrong | 4 |

The TOON swap changes which individual samples fail, but the errors are balanced in aggregate.

### Flips Against TOON

| question_id | outcome | raw prediction | TOON prediction | paper chars |
| --- | --- | --- | --- | ---: |
| `6PcJEFKvBD` | reject | Reject | Accept | 43,892 |
| `Woiqqi5bYV` | accept | Accept | Reject | 106,999 |
| `wxEASOHHdT` | reject | Reject | Accept | 32,714 |
| `j1FLTvgyAh` | reject | Reject | Accept | 50,106 |
| `EmxpDiPgRu` | accept | Accept | Reject | 106,988 |
| `xawA8X5dHq` | reject | Reject | Accept | 45,092 |

### Flips In Favor Of TOON

| question_id | outcome | raw prediction | TOON prediction | paper chars |
| --- | --- | --- | --- | ---: |
| `wWcNhS4g1U` | reject | Accept | Reject | 98,219 |
| `xAYOfMV264` | reject | Accept | Reject | 51,475 |
| `fAAaT826Vv` | accept | Reject | Accept | 82,327 |
| `zd5Knrtja4` | reject | Accept | Reject | 59,941 |
| `xRDYDI6Rc9` | reject | Accept | Reject | 95,004 |
| `waIltEWDr8` | reject | Accept | Reject | 55,573 |

## Stratified Checks

### By Outcome

| outcome | raw correct | TOON correct |
| --- | ---: | ---: |
| accept | 29/30 | 28/30 |
| reject | 21/30 | 22/30 |

### By Paper Length

| length bucket | chars | raw correct | TOON correct |
| --- | --- | ---: | ---: |
| short | 16,985-54,296 | 18/20 | 15/20 |
| medium | 54,870-81,296 | 16/20 | 18/20 |
| long | 82,327-171,230 | 16/20 | 17/20 |

## Interpretation

The swap is not damaging in aggregate on the controlled validation set. Across the three controlled comparisons, TOON preserves accuracy within 0 to 1 sample while reducing total tokens by about 67-68%.

The main residual risk is distributional: the 60-sample run shows a weaker short-paper bucket for TOON, offset by medium and long papers. If this needs publication-grade confidence, the next experiment should scale the controlled pair to 100+ samples and report confidence intervals plus the same outcome and length stratification.

## Artifacts

- `runs/raw_neutral_vs_retrieval_toon.md`
- `runs/raw_neutral_repeat1_vs_retrieval_toon_repeat1.md`
- `runs/raw_neutral_60_vs_retrieval_toon_60.md`
- `domains/paper_review/dataset_eval_60.csv`
