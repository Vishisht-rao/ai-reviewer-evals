# OpenAI Evaluator (GPT-5.5)

Submits TMLR papers to GPT-5.5 via the OpenAI Batch API and downloads structured reviews.

## Setup

```bash
pip install openai pymupdf
```

Set `API_KEY` at the top of each script. The prompt used is `prompt_gpt_5_5.txt`.

## Batch Run (full directory)

```bash
# Step 1 — submit
python3 openai_evals/run_openai_batch.py submit \
  --pdf-dir <category>/pdfs \
  --output-dir <category>/output_gpt55_v3

# Step 2 — poll until complete (checks every 60s)
python3 openai_evals/run_openai_batch.py poll \
  --output-dir <category>/output_gpt55_v3
```

Submit options:
- `--extract-text` — send extracted plain text instead of raw PDF (much cheaper; requires pymupdf)
- `--paper-list <file>` — only submit papers listed in a text file (one stem per line)
- `--sample K` — randomly sample K PDFs from the directory
- `--prompt-file <path>` — use a different prompt file
- `--batch-id-file <path>` — override default `batch_id.txt` location

Poll options:
- `--interval <seconds>` — polling interval (default: 60)
- `--batch-id-file <path>` — override batch ID file location

## Test Run (small sanity check, 2 papers per category)

```bash
python3 openai_evals/run_openai_test.py --output-dir openai_evals/test_output
```

Options: `--seed`, `--extract-text`, `--categories <names...>`, `--paper-list <file>`

## Notes

- Each PDF is uploaded to the OpenAI Files API (`purpose="user_data"`), referenced by `file_id` in the batch JSONL — keeps the payload small regardless of PDF count.
- Uploaded file IDs are saved to `<output-dir>/batch_id_file_ids.json`. Delete them manually when done to avoid storage charges.
- Papers in `prompt_examples_exclusion.json` are automatically excluded from all runs.
- Model and reasoning effort are set as constants at the top of each script (`MODEL`, `REASONING_EFFORT`).
