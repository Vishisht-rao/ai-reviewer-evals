# TMLR AI Reviewer Evaluations

Evaluation framework for benchmarking AI paper reviewers against ground-truth accept/reject decisions from TMLR (Transactions on Machine Learning Research).

## Overview

We evaluate four AI reviewing systems on a curated set of TMLR papers across five categories:

| Category | Papers | GT Claims |
|---|---|---|
| `accepted` | ~50 | Yes |
| `both_no` | ~50 | No |
| `claims_only_no` | ~50 | No |
| `flaws_gemini` | ~67 | No |
| `flaws_openai` | ~50 | No |

Ground truth is sourced from TMLR Action Editor decisions. The `flaws_*` categories contain papers rejected specifically due to unsupported claims; for these, we also annotate *where* the AI reviewer located the flaw.

## Reviewers

| Folder | System | Model |
|---|---|---|
| `openai_evals/` | Custom prompt | GPT-5.5 (`prompt_gpt_5_5.txt`) |
| `claude_evals/` | Custom prompt | Claude Opus 4.7 (`prompt_opus_4_7.txt`) |
| `cspaper_evals/` | [CSPaper](https://cspaper.io) agentic reviewer | via CSPaper API |
| `reviewertoo_evals/` | [ReviewerToo](https://reviewertoo.org) | Internal |

## Repository Structure

```
ai-reviewer-evals/
├── openai_evals/          GPT-5.5 batch + test scripts and prompt
├── claude_evals/          Claude Opus 4.7 batch + test scripts and prompt
├── cspaper_evals/         CSPaper submission scripts
├── reviewertoo_evals/     ReviewerToo submission scripts and raw outputs
├── create_annotation_csv.py   Generate annotation CSVs from output dirs
└── tmlr_ai_reviewer_eval.ipynb  Analysis notebook (accuracy, distributions)
```

## Test Cases and Outputs

All PDFs, ground truths, and reviewer outputs are distributed as a separate archive (not tracked in this repo due to size):

> **Download:** [Google Drive — test_cases_all.zip](#https://drive.google.com/drive/folders/1l0PfN0C-jjXlT-qECT7C_EDHT61o2o61?usp=sharing)

The archive has one subfolder per category, each containing:
- `pdfs/` — input papers
- `ground_truths/` (or `ground_truth_errors/` for FLAWS) — AE decisions
- `output_cspaper/`, `output_gpt55_v3/`, `output_opus47/`, `output_reviewertoo/` — reviewer outputs

Point `tmlr_ai_reviewer_eval.ipynb` at the extracted archive by setting `BASE_DIR` in cell 1.

## Quick Start

### GPT-5.5 (OpenAI Batch API)

```bash
# Set your API key at the top of the script
# Submit a full category
python3 openai_evals/run_openai_batch.py submit \
  --pdf-dir <category>/pdfs \
  --output-dir <category>/output_gpt55_v3

# Poll until complete, then download results
python3 openai_evals/run_openai_batch.py poll \
  --output-dir <category>/output_gpt55_v3
```

See [openai_evals/README.md](openai_evals/README.md) for full options.

### Claude Opus 4.7 (Anthropic Batch API)

```bash
python3 claude_evals/run_claude_batch.py submit \
  --pdf-dir <category>/pdfs \
  --output-dir <category>/output_opus47

python3 claude_evals/run_claude_batch.py poll \
  --output-dir <category>/output_opus47
```

See [claude_evals/README.md](claude_evals/README.md) for full options.

### CSPaper

See [cspaper_evals/README.md](cspaper_evals/README.md).

### ReviewerToo

See [reviewertoo_evals/README.md](reviewertoo_evals/README.md).

## Exclusions

Papers used as calibration examples in the GPT-5.5 and Opus 4.7 prompts are listed in `openai_evals/prompt_examples_exclusion.json` and `claude_evals/prompt_examples_exclusion.json` (identical content). These papers are automatically excluded from all evaluations by the batch and test scripts.

## Analysis

1. Download and extract `test_cases_all.zip` so that the `test_cases_all/` folder sits **next to this notebook** inside `ai-reviewer-evals/`:
   ```
   ai-reviewer-evals/
   ├── tmlr_ai_reviewer_eval.ipynb
   └── test_cases_all/
       ├── accepted/
       ├── both_no/
       └── ...
   ```
2. Open `tmlr_ai_reviewer_eval.ipynb` and run all cells. `BASE_DIR` is already set to `Path("test_cases_all")` — no changes needed.

The notebook produces:
- Per-category claims distributions
- Claims accuracy per reviewer
- Eligible + common paper comparisons across GPT-5.5, Opus 4.7, and CSPaper
- FLAWS annotation summaries
