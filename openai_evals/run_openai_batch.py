"""
Batch API evaluation script for TMLR paper reviews using OpenAI.

Submits PDFs for review via the OpenAI Batch API (async, 50% cost discount).
By default, each PDF is uploaded via the Files API (purpose="user_data") and referenced
by file_id in the batch JSONL — this keeps the JSONL small regardless of PDF size.
With --extract-text, plain text is extracted from each PDF and sent inline instead
(requires pymupdf; much cheaper but GPT cannot see figures/tables).

Usage:
    python run_openai_batch.py submit --pdf-dir <dir> --output-dir <dir> [options]
    python run_openai_batch.py poll   --output-dir <dir> [--batch-id-file <path>] [--interval <secs>]

Submit options:
    --batch-id-file <path>   Where to write the batch ID (default: <output-dir>/batch_id.txt)
    --extract-text           Extract plain text from PDFs instead of uploading via Files API
    --paper-list <file>      Text file with one paper stem per line; only those papers are
                             submitted (useful for re-running a subset that got wrong answers)
    --sample K               Randomly sample K PDFs from the directory (after exclusion/paper-list
                             filtering); selected stems saved to <output-dir>/sampled_papers.txt
    --prompt-file <path>     Path to the prompt text file (default: prompt_v1.txt next to this script)

The batch-id-file defaults to <output-dir>/batch_id.txt.
Poll interval defaults to 60 seconds.
Output reviews are saved as <output-dir>/<pdf-stem>.md.
Uploaded file IDs are saved to <output-dir>/uploaded_file_ids.json (for manual cleanup).
"""

import argparse
import base64
import hashlib
import io
import json
import random
import re
import sys
import time
from pathlib import Path

import openai

API_KEY           = "your-openai-api-key"
MODEL             = "gpt-5.5"
MAX_OUTPUT_TOKENS = 8192
REASONING_EFFORT  = "xhigh"  # "none", "low", "medium", "high", "xhigh"
PROMPT_FILE       = Path(__file__).parent / "prompt_v3.txt"
EXCLUSION_FILE    = Path(__file__).parent / "prompt_examples_exclusion.json"


def pdf_to_text(path: Path) -> str:
    try:
        import fitz
    except ImportError:
        sys.exit("ERROR: pymupdf not installed. Run: pip install pymupdf")
    doc = fitz.open(str(path))
    return "\n".join(page.get_text() for page in doc)


def make_content_text(pdf_path: Path, prompt_text: str) -> list:
    return [
        {"type": "input_text", "text": pdf_to_text(pdf_path)},
        {"type": "input_text", "text": prompt_text},
    ]


def make_content_file_id(file_id: str, prompt_text: str) -> list:
    return [
        {"type": "input_file", "file_id": file_id},
        {"type": "input_text", "text": prompt_text},
    ]


def make_content_inline(pdf_path: Path, prompt_text: str) -> list:
    """Inline base64 — only suitable for the synchronous Responses API, not batch."""
    pdf_b64 = base64.b64encode(pdf_path.read_bytes()).decode("utf-8")
    return [
        {
            "type": "input_file",
            "filename": pdf_path.name,
            "file_data": f"data:application/pdf;base64,{pdf_b64}",
        },
        {"type": "input_text", "text": prompt_text},
    ]


def load_prompt(prompt_file: Path) -> str:
    if not prompt_file.exists():
        sys.exit(f"ERROR: prompt file not found at {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")


def load_exclusions() -> dict:
    if not EXCLUSION_FILE.exists():
        return {}
    return json.loads(EXCLUSION_FILE.read_text(encoding="utf-8")).get("excluded_papers", {})


def make_custom_id(stem: str) -> str:
    """Return a custom_id safe for the API and at most 64 characters."""
    safe = re.sub(r"[^A-Za-z0-9_\-]", "_", stem)
    if len(safe) <= 64:
        return safe
    h = hashlib.md5(stem.encode()).hexdigest()[:8]
    return safe[:55] + "_" + h


# ── submit ──────────────────────────────────────────────────────────────────────

def cmd_submit(args: argparse.Namespace) -> None:
    client = openai.OpenAI(api_key=API_KEY)
    prompt_file = Path(args.prompt_file) if args.prompt_file else PROMPT_FILE
    prompt_text = load_prompt(prompt_file)

    pdf_dir    = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_id_file = Path(args.batch_id_file) if args.batch_id_file else output_dir / "batch_id.txt"
    id_map_file   = batch_id_file.with_name(batch_id_file.stem + "_id_map.json")

    exclusions    = load_exclusions()
    exclusion_key = pdf_dir.parent.name
    excluded      = set(exclusions.get(exclusion_key, []))
    if excluded:
        print(f"Excluding {len(excluded)} paper(s) listed in {EXCLUSION_FILE.name} (key: {exclusion_key})")

    paper_list_filter = None
    if args.paper_list:
        paper_list_path = Path(args.paper_list)
        if not paper_list_path.exists():
            sys.exit(f"ERROR: paper list file not found at {paper_list_path}")
        paper_list_filter = set(
            line.strip() for line in paper_list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        print(f"Paper list filter: {len(paper_list_filter)} stem(s) from {paper_list_path.name}")

    all_pdfs = sorted(pdf_dir.glob("*.pdf"))
    pdfs = [p for p in all_pdfs if p.stem not in excluded]
    skipped = len(all_pdfs) - len(pdfs)
    if skipped:
        print(f"Skipped {skipped} excluded PDF(s).")

    if paper_list_filter is not None:
        before = len(pdfs)
        pdfs = [p for p in pdfs if p.stem in paper_list_filter]
        print(f"Paper list filter: kept {len(pdfs)} of {before} non-excluded PDF(s).")
    if not pdfs:
        sys.exit(f"No PDF files found in {pdf_dir} after exclusion filtering")

    if args.sample is not None:
        k = args.sample
        if k > len(pdfs):
            print(f"WARNING: --sample {k} exceeds available PDFs ({len(pdfs)}); using all.")
            k = len(pdfs)
        pdfs = random.sample(pdfs, k)
        pdfs.sort()
        sampled_list_file = output_dir / "sampled_papers.txt"
        sampled_list_file.write_text("\n".join(p.stem for p in pdfs) + "\n", encoding="utf-8")
        print(f"Sampled {k} PDF(s); stems saved -> {sampled_list_file}")

    if args.extract_text:
        print(f"Extracting text from {len(pdfs)} PDF(s) (--extract-text mode)...")
    else:
        print(f"Uploading {len(pdfs)} PDF(s) via Files API (purpose='user_data')...")

    batch_requests = []
    id_map = {}
    file_id_map = {}  # stem -> OpenAI file_id (saved for manual cleanup)
    for pdf_path in pdfs:
        stem      = pdf_path.stem
        custom_id = make_custom_id(stem)
        id_map[custom_id] = stem
        print(f"  {stem} ...", end=" ", flush=True)

        if args.extract_text:
            content = make_content_text(pdf_path, prompt_text)
            print("extracted")
        else:
            with pdf_path.open("rb") as fh:
                uploaded_file = client.files.create(
                    file=(pdf_path.name, fh, "application/pdf"),
                    purpose="user_data",
                )
            file_id_map[stem] = uploaded_file.id
            content = make_content_file_id(uploaded_file.id, prompt_text)
            print(f"uploaded ({uploaded_file.id})")

        batch_requests.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": MODEL,
                "input": [{"role": "user", "content": content}],
                "reasoning": {"effort": REASONING_EFFORT},
                "max_output_tokens": MAX_OUTPUT_TOKENS,
            },
        })

    id_map_file.write_text(json.dumps(id_map, indent=2), encoding="utf-8")
    if file_id_map:
        file_id_map_file = batch_id_file.with_name(batch_id_file.stem + "_file_ids.json")
        file_id_map_file.write_text(json.dumps(file_id_map, indent=2), encoding="utf-8")
        print(f"\nFile IDs saved  -> {file_id_map_file}  (delete manually when done)")

    jsonl_bytes = "\n".join(json.dumps(r) for r in batch_requests).encode("utf-8")
    print(f"\nUploading batch JSONL ({len(jsonl_bytes) // 1024} KB)...")
    uploaded_jsonl = client.files.create(
        file=("batch_requests.jsonl", io.BytesIO(jsonl_bytes), "application/json"),
        purpose="batch",
    )

    print(f"Creating batch ({len(batch_requests)} requests)...")
    batch = client.batches.create(
        input_file_id=uploaded_jsonl.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    batch_id_file.write_text(batch.id, encoding="utf-8")
    print(f"Batch submitted:  {batch.id}")
    print(f"Batch ID saved -> {batch_id_file}")
    print(f"ID map saved   -> {id_map_file}")


# ── poll ────────────────────────────────────────────────────────────────────────

def cmd_poll(args: argparse.Namespace) -> None:
    client = openai.OpenAI(api_key=API_KEY)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_id_file = Path(args.batch_id_file) if args.batch_id_file else output_dir / "batch_id.txt"
    if not batch_id_file.exists():
        sys.exit(f"ERROR: batch ID file not found at {batch_id_file}")

    batch_id = batch_id_file.read_text(encoding="utf-8").strip()
    interval = args.interval

    id_map_file = batch_id_file.with_name(batch_id_file.stem + "_id_map.json")
    id_map = json.loads(id_map_file.read_text(encoding="utf-8")) if id_map_file.exists() else {}

    print(f"Polling batch {batch_id} every {interval}s ...")
    terminal_statuses = {"completed", "failed", "expired", "cancelled"}
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"  [{batch.status}]  "
            f"total={counts.total}  completed={counts.completed}  failed={counts.failed}"
        )
        if batch.status in terminal_statuses:
            break
        time.sleep(interval)

    if batch.status != "completed":
        sys.exit(f"Batch ended with status: {batch.status}")

    print("\nBatch completed — downloading results...")
    file_content = client.files.content(batch.output_file_id).text
    token_rows, errors = [], []

    for line in file_content.splitlines():
        if not line.strip():
            continue
        item      = json.loads(line)
        custom_id = item["custom_id"]
        stem      = id_map.get(custom_id, custom_id)

        if item.get("error"):
            errors.append((stem, str(item["error"])))
            print(f"  [FAIL]  {stem}: {item['error']}")
            continue

        resp = item.get("response", {})
        if resp.get("status_code") != 200:
            errors.append((stem, f"status_code={resp.get('status_code')}"))
            print(f"  [FAIL]  {stem}: status_code={resp.get('status_code')}")
            continue

        body = resp["body"]
        text = None
        for out_item in body.get("output", []):
            for part in out_item.get("content", []):
                if part.get("type") == "output_text":
                    text = part["text"]
                    break
            if text:
                break
        if text is None:
            text = body.get("output_text")
        if text is None:
            errors.append((stem, "no_text_found"))
            print(f"  [FAIL]  {stem}: no text found in response")
            continue

        usage   = body.get("usage", {})
        in_tok  = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        out_path = output_dir / f"{stem}.md"
        out_path.write_text(text, encoding="utf-8")
        token_rows.append((stem, in_tok, out_tok))
        print(f"  [saved] {stem}.md  (in={in_tok:,}  out={out_tok:,})")

    if errors:
        print(f"\n{len(errors)} failed request(s):")
        for stem, err in errors:
            print(f"  {stem}: {err}")

    if token_rows:
        total_in  = sum(r[1] for r in token_rows)
        total_out = sum(r[2] for r in token_rows)
        n = len(token_rows)
        print(f"\nToken summary ({n} paper(s)):")
        print(f"  Input  — total: {total_in:>10,}   avg: {total_in // n:>8,}")
        print(f"  Output — total: {total_out:>10,}   avg: {total_out // n:>8,}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TMLR batch review evaluator (OpenAI Batch API)")
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("submit", help="Upload PDFs and submit a batch job")
    sp.add_argument("--pdf-dir",       required=True,       help="Directory containing PDF files to review")
    sp.add_argument("--output-dir",    required=True,       help="Directory where completed reviews are saved")
    sp.add_argument("--batch-id-file", default=None,        help="Path to write the batch ID (default: <output-dir>/batch_id.txt)")
    sp.add_argument("--extract-text",  action="store_true", help="Extract plain text from PDFs instead of base64 encoding (requires pymupdf; much cheaper)")
    sp.add_argument("--paper-list",    default=None,        help="Text file with one paper stem per line; only those papers are submitted")
    sp.add_argument("--sample",        type=int, default=None, metavar="K", help="Randomly sample K PDFs; stems saved to <output-dir>/sampled_papers.txt")
    sp.add_argument("--prompt-file",   default=None,        help="Path to prompt text file (default: prompt_v1.txt next to this script)")

    pp = sub.add_parser("poll", help="Poll a submitted batch and save completed reviews")
    pp.add_argument("--output-dir",    required=True, help="Directory where completed reviews are saved")
    pp.add_argument("--batch-id-file", default=None,  help="Path to the batch ID file (default: <output-dir>/batch_id.txt)")
    pp.add_argument("--interval",      type=int, default=60, help="Polling interval in seconds (default: 60)")

    args = parser.parse_args()
    if args.command == "submit":
        cmd_submit(args)
    else:
        cmd_poll(args)


if __name__ == "__main__":
    main()
