"""
Batch API evaluation script for TMLR paper reviews.

Submit PDFs for review via the Anthropic Batch API (async, billed at 50% discount).
By default, PDFs are base64-encoded and sent inline (works reliably with the batch API).
With --extract-text, plain text is extracted from each PDF and sent inline instead
(requires pymupdf; much cheaper but Claude cannot see figures/tables).

Usage:
    python run_claude_batch.py submit --pdf-dir <dir> --output-dir <dir> [options]
    python run_claude_batch.py poll   --output-dir <dir> [--batch-id-file <path>] [--interval <secs>]

Submit options:
    --batch-id-file <path>   Where to write the batch ID (default: <output-dir>/batch_id.txt)
    --extract-text           Extract plain text from PDFs instead of base64 encoding
    --paper-list <file>      Text file with one paper stem per line; only those papers are
                             submitted (useful for re-running a subset that got wrong answers)
    --sample K               Randomly sample K PDFs from the directory (after exclusion/paper-list
                             filtering); selected stems saved to <output-dir>/sampled_papers.txt
    --prompt-file <path>     Path to the prompt text file (default: prompt_opus_4_7.txt next to this script)

The batch-id-file defaults to <output-dir>/batch_id.txt.
Poll interval defaults to 60 seconds.
Output reviews are saved as <output-dir>/<pdf-stem>.md.
"""

import argparse
import base64
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

import anthropic

API_KEY                = "your-anthropic-api-key"
MODEL                  = "claude-opus-4-7"
MAX_TOKENS      = 8192
ENABLE_THINKING = False
THINKING_EFFORT = "high"  # "low", "medium", "high", "xhigh", "max" — only used when ENABLE_THINKING = True
PROMPT_FILE            = Path(__file__).parent / "prompt_opus_4_7.txt"
EXCLUSION_FILE         = Path(__file__).parent / "prompt_examples_exclusion.json"


def pdf_to_text(path: Path) -> str:
    try:
        import fitz
    except ImportError:
        sys.exit("ERROR: pymupdf not installed. Run: pip install pymupdf")
    doc = fitz.open(str(path))
    return "\n".join(page.get_text() for page in doc)


def load_prompt() -> str:
    if not PROMPT_FILE.exists():
        sys.exit(f"ERROR: prompt file not found at {PROMPT_FILE}")
    return PROMPT_FILE.read_text(encoding="utf-8")


def load_exclusions() -> dict:
    if not EXCLUSION_FILE.exists():
        return {}
    return json.loads(EXCLUSION_FILE.read_text(encoding="utf-8")).get("excluded_papers", {})


def make_custom_id(stem: str) -> str:
    """Return a custom_id that is safe and at most 64 characters."""
    safe = re.sub(r"[^A-Za-z0-9_\-]", "_", stem)
    if len(safe) <= 64:
        return safe
    # Truncate to 55 chars + "_" + 8-char hash of original stem
    h = hashlib.md5(stem.encode()).hexdigest()[:8]
    return safe[:55] + "_" + h


# ── submit ──────────────────────────────────────────────────────────────────────

def cmd_submit(args: argparse.Namespace) -> None:
    client = anthropic.Anthropic(api_key=API_KEY)
    prompt_file = Path(args.prompt_file) if args.prompt_file else PROMPT_FILE
    if not prompt_file.exists():
        sys.exit(f"ERROR: prompt file not found at {prompt_file}")
    prompt_text = prompt_file.read_text(encoding="utf-8")

    pdf_dir    = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_id_file = Path(args.batch_id_file) if args.batch_id_file else output_dir / "batch_id.txt"
    # id_map is named after the batch_id_file to avoid overwriting a previous batch's map
    id_map_file   = batch_id_file.with_name(batch_id_file.stem + "_id_map.json")

    # Load exclusions — key is the parent directory name of pdf_dir
    exclusions    = load_exclusions()
    exclusion_key = pdf_dir.parent.name
    excluded      = set(exclusions.get(exclusion_key, []))
    if excluded:
        print(f"Excluding {len(excluded)} paper(s) listed in {EXCLUSION_FILE.name} (key: {exclusion_key})")

    # Load optional paper-list filter
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
        pdfs.sort()  # deterministic order for readability
        sampled_list_file = output_dir / "sampled_papers.txt"
        sampled_list_file.write_text(
            "\n".join(p.stem for p in pdfs) + "\n", encoding="utf-8"
        )
        print(f"Sampled {k} PDF(s); stems saved → {sampled_list_file}")

    if args.extract_text:
        print(f"Extracting text from {len(pdfs)} PDF(s) (--extract-text mode)...")
    else:
        print(f"Base64-encoding {len(pdfs)} PDF(s) for inline submission...")

    requests = []
    id_map   = {}  # custom_id -> original stem (for poll to recover filenames)
    for pdf_path in pdfs:
        stem      = pdf_path.stem
        custom_id = make_custom_id(stem)
        id_map[custom_id] = stem
        print(f"  {stem} ...", end=" ", flush=True)

        if args.extract_text:
            paper_block = {"type": "text", "text": pdf_to_text(pdf_path)}
            print("extracted")
        else:
            pdf_b64 = base64.b64encode(pdf_path.read_bytes()).decode("utf-8")
            paper_block = {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_b64},
            }
            print(f"encoded ({len(pdf_b64) // 1024} KB)")

        params = {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": [paper_block, {"type": "text", "text": prompt_text}],
                }
            ],
        }
        if ENABLE_THINKING:
            params["thinking"] = {"type": "adaptive"}
            params["output_config"] = {"effort": THINKING_EFFORT}  # passed directly in batch params JSON
        requests.append({"custom_id": custom_id, "params": params})

    # Save the id map so poll can recover original filenames
    id_map_file.write_text(json.dumps(id_map, indent=2), encoding="utf-8")

    print(f"\nCreating batch ({len(requests)} requests)...")
    batch = client.messages.batches.create(requests=requests)
    batch_id_file.write_text(batch.id, encoding="utf-8")
    print(f"Batch submitted:  {batch.id}")
    print(f"Batch ID saved →  {batch_id_file}")
    print(f"ID map saved   →  {id_map_file}")


# ── poll ────────────────────────────────────────────────────────────────────────

def cmd_poll(args: argparse.Namespace) -> None:
    client = anthropic.Anthropic(api_key=API_KEY)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_id_file = Path(args.batch_id_file) if args.batch_id_file else output_dir / "batch_id.txt"
    if not batch_id_file.exists():
        sys.exit(f"ERROR: batch ID file not found at {batch_id_file}")

    batch_id = batch_id_file.read_text(encoding="utf-8").strip()
    interval = args.interval

    # Load id map paired with this batch_id_file (same stem, _id_map.json suffix)
    id_map_file = batch_id_file.with_name(batch_id_file.stem + "_id_map.json")
    id_map = json.loads(id_map_file.read_text(encoding="utf-8")) if id_map_file.exists() else {}

    print(f"Polling batch {batch_id} every {interval}s ...")
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"  [{batch.processing_status}]  "
            f"processing={counts.processing}  succeeded={counts.succeeded}  "
            f"errored={counts.errored}  expired={counts.expired}  canceled={counts.canceled}"
        )
        if batch.processing_status == "ended":
            break
        time.sleep(interval)

    print("\nBatch ended — saving results...")
    token_rows, errors = [], []

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        stem      = id_map.get(custom_id, custom_id)  # fall back to custom_id if no map
        if result.result.type == "succeeded":
            text = next((block.text for block in result.result.message.content if block.type == "text"), None)
            if text is None:
                errors.append((stem, "no_text_block"))
                print(f"  [FAIL]  {stem}: succeeded but contained no text block")
                continue
            usage = result.result.message.usage
            out_path = output_dir / f"{stem}.md"
            out_path.write_text(text, encoding="utf-8")
            token_rows.append((stem, usage.input_tokens, usage.output_tokens))
            print(f"  [saved] {stem}.md  (in={usage.input_tokens:,}  out={usage.output_tokens:,})")
        else:
            error_detail = getattr(result.result, "error", None)
            error_msg = f"{result.result.type}"
            if error_detail:
                error_msg += f" | {getattr(error_detail, 'type', '')} | {getattr(error_detail, 'message', error_detail)}"
            errors.append((stem, error_msg))
            print(f"  [FAIL]  {stem}: {error_msg}")

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
    parser = argparse.ArgumentParser(description="TMLR batch review evaluator (Anthropic Batch API)")
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("submit", help="Upload PDFs and submit a batch job")
    sp.add_argument("--pdf-dir",       required=True,       help="Directory containing PDF files to review")
    sp.add_argument("--output-dir",    required=True,       help="Directory where completed reviews are saved")
    sp.add_argument("--batch-id-file", default=None,        help="Path to write the batch ID (default: <output-dir>/batch_id.txt)")
    sp.add_argument("--extract-text",  action="store_true", help="Extract plain text from PDFs instead of uploading via Files API (requires pymupdf; much cheaper)")
    sp.add_argument("--paper-list",    default=None,        help="Text file with one paper stem per line; only those papers are submitted (for targeted re-runs)")
    sp.add_argument("--sample",        type=int, default=None, metavar="K", help="Randomly sample K PDFs from the directory (after exclusion/paper-list filtering); selected stems saved to <output-dir>/sampled_papers.txt")
    sp.add_argument("--prompt-file",   default=None, help="Path to the prompt text file (default: prompt_v2.txt next to this script)")

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
