# ReviewerToo Evaluations

Scripts and raw outputs for running TMLR papers through [ReviewerToo](https://reviewertoo.org).

## Setup

```bash
pip install httpx tqdm
```

You need a ReviewerToo API key. Set it via environment variable or pass it on the command line:

```bash
export REVIEWERTOO_API_KEY="your-api-key"
export REVIEWERTOO_API_URL="https://reviewertoo.org"
```

## Usage

ReviewerToo is run via `reviewertoo_client.py` (from the ReviewerToo project — not included here). Use the `recover_results.py` script in this directory to fetch completed reviews from the API.

```bash
# Recover results for a batch of papers
python3 reviewertoo_evals/recover_results.py \
  --api-url https://reviewertoo.org \
  --api-key <your-key> \
  --output <output-dir>
```

## Raw Outputs

`recovered/` contains the raw ReviewerToo JSON output for each paper, with one subdirectory per submission (named `{paper_title}_{task_id}`). Each subdirectory contains:
- `metareview.json` — aggregated meta-review with final Yes/No verdicts
- `reviews/Reviewer_N.json` — individual reviewer outputs

These raw outputs were converted to `.md` stub files (matching the format expected by `tmlr_ai_reviewer_eval.ipynb`) using `convert_reviewertoo.py`.
