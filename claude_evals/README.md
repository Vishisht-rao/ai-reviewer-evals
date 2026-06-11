# Claude Evaluator (Opus 4.7)

Submits TMLR papers to Claude Opus 4.7 via the Anthropic Batch API and downloads structured reviews.

## Setup

```bash
pip install anthropic pymupdf
```

Set `API_KEY` at the top of each script. The prompt used is `prompt_opus_4_7.txt`.

## Batch Run (full directory)

```bash
# Step 1 — submit
python3 claude_evals/run_claude_batch.py submit \
  --pdf-dir <category>/pdfs \
  --output-dir <category>/output_opus47

# Step 2 — poll until complete (checks every 60s)
python3 claude_evals/run_claude_batch.py poll \
  --output-dir <category>/output_opus47
```

Submit options:
- `--extract-text` — send extracted plain text instead of raw PDF (much cheaper; requires pymupdf)
- `--paper-list <file>` — only submit papers listed in a text file (one stem per line)
- `--batch-id-file <path>` — override default `batch_id.txt` location
- `--prompt-file <path>` — use a different prompt file

Poll options:
- `--interval <seconds>` — polling interval (default: 60)
- `--batch-id-file <path>` — override batch ID file location

## Test Run (small sanity check, 2 papers per category)

```bash
python3 claude_evals/run_claude_test.py --output-dir claude_evals/test_output
```

Options: `--seed`, `--extract-text`, `--categories <names...>`

## Notes

- PDFs are base64-encoded and embedded inline in each batch request (no separate Files API step).
- Use `--extract-text` for large batches to reduce token cost significantly (~10× cheaper; Claude cannot see figures in text mode).
- Papers in `prompt_examples_exclusion.json` are automatically excluded from all runs.
- Model and token limits are set as constants at the top of each script (`MODEL`, `MAX_TOKENS`).
