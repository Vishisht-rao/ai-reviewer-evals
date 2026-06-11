"""
Test evaluation script for TMLR paper reviews using regular (non-batch) API calls.

Selects 2 PDFs at random from each of 7 source categories, excluding papers that were
used as calibration examples in the prompt (listed in prompt_examples_exclusion.json).

Outputs:
  <output-dir>/selected_pdfs/<category>/<paper>.pdf  — copies of the selected PDFs
  <output-dir>/reviews/<category>/<paper>.md         — Claude's review

Usage:
    python run_claude_test.py [--output-dir <dir>] [--seed <int>] [--extract-text]

--extract-text  Send extracted plain text instead of the raw PDF (requires pymupdf).
                Much cheaper (~10x fewer tokens) but Claude cannot see figures/tables.
"""

import argparse
import base64
import json
import random
import shutil
import sys
from pathlib import Path

import anthropic

API_KEY               = "your-anthropic-api-key"
MODEL                 = "claude-opus-4-7"
MAX_TOKENS       = 8192
ENABLE_THINKING  = True
THINKING_EFFORT  = "high"  # "low", "medium", "high", "xhigh", "max" — only used when ENABLE_THINKING = True
PROMPT_FILE           = Path(__file__).parent / "prompt_opus_4_7.txt"
EXCLUSION_FILE = Path(__file__).parent / "prompt_examples_exclusion.json"
BASE_DIR       = Path(__file__).parent.parent  # repo root

# 2 PDFs sampled from each entry.  "flaws" pools gemini + openai PDFs together.
TEST_SOURCES = [
    {
        "name": "accepted",
        "pdf_dir": "tmlr_test_cases_final_accepted/accepted_TMLR",
        "exclusion_key": "tmlr_test_cases_final_accepted",
    },
    {
        "name": "audience_only_no",
        "pdf_dir": "tmlr_test_cases_final_audience_only_no/rejected_TMLR_audience_only",
        "exclusion_key": "tmlr_test_cases_final_audience_only_no",
    },
    {
        "name": "both_no",
        "pdf_dir": "tmlr_test_cases_final_both_no/rejected_TMLR_both_no",
        "exclusion_key": "tmlr_test_cases_final_both_no",
    },
    {
        "name": "claims_only_no",
        "pdf_dir": "tmlr_test_cases_final_claims_only_no/rejected_TMLR_claims_only",
        "exclusion_key": "tmlr_test_cases_final_claims_only_no",
    },
    {
        "name": "class_reports",
        "pdf_dir": "test_cases_no_interest_TMLR/test_cases_class_reports/class_reports",
        "exclusion_key": None,
    },
    {
        "name": "non_ml",
        "pdf_dir": "test_cases_no_interest_TMLR/test_cases_non_ml/non_ml",
        "exclusion_key": None,
    },
    {
        "name": "flaws",
        "pdf_dirs": [
            "FLAWS_test_cases/gemini/pdfs",
            "FLAWS_test_cases/openai/pdfs",
        ],
        "exclusion_key": None,
    },
]


def pdf_to_text(path: Path) -> str:
    try:
        import fitz
    except ImportError:
        sys.exit("ERROR: pymupdf not installed. Run: pip install pymupdf")
    doc = fitz.open(str(path))
    return "\n".join(page.get_text() for page in doc)


def make_content(pdf_path: Path, prompt_text: str, extract_text: bool) -> list:
    if extract_text:
        paper_block = {"type": "text", "text": pdf_to_text(pdf_path)}
    else:
        pdf_b64 = base64.standard_b64encode(pdf_path.read_bytes()).decode("utf-8")
        paper_block = {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_b64},
        }
    return [paper_block, {"type": "text", "text": prompt_text}]


def load_prompt() -> str:
    if not PROMPT_FILE.exists():
        sys.exit(f"ERROR: prompt file not found at {PROMPT_FILE}")
    return PROMPT_FILE.read_text(encoding="utf-8")


def load_exclusions() -> dict:
    if not EXCLUSION_FILE.exists():
        return {}
    return json.loads(EXCLUSION_FILE.read_text(encoding="utf-8")).get("excluded_papers", {})


def collect_pdfs(source: dict) -> list[Path]:
    if "pdf_dirs" in source:
        all_pdfs = []
        for rel in source["pdf_dirs"]:
            d = BASE_DIR / rel
            if d.exists():
                all_pdfs.extend(sorted(d.glob("*.pdf")))
        return all_pdfs
    d = BASE_DIR / source["pdf_dir"]
    return sorted(d.glob("*.pdf")) if d.exists() else []


def sample_pdfs(source: dict, exclusions: dict, n: int = 2) -> list[Path]:
    excluded = set(exclusions.get(source.get("exclusion_key") or "", []))
    valid = [p for p in collect_pdfs(source) if p.stem not in excluded]
    if len(valid) < n:
        print(f"  WARNING [{source['name']}]: only {len(valid)} valid PDF(s), need {n}", file=sys.stderr)
    return random.sample(valid, min(n, len(valid)))


def main() -> None:
    parser = argparse.ArgumentParser(description="TMLR sanity-check reviewer (regular API, fixed test set)")
    valid_categories = [s["name"] for s in TEST_SOURCES]
    parser.add_argument("--output-dir",    default="./test_output", help="Output directory (default: ./test_output)")
    parser.add_argument("--seed",          type=int, default=42,    help="Random seed for PDF sampling (default: 42)")
    parser.add_argument("--extract-text",  action="store_true",     help="Extract plain text from PDFs instead of sending raw PDF (requires pymupdf; much cheaper)")
    parser.add_argument("--categories",    nargs="+", metavar="CAT", help=f"Only run these categories (default: all). Choices: {', '.join(valid_categories)}")
    args = parser.parse_args()

    if args.categories:
        unknown = set(args.categories) - set(valid_categories)
        if unknown:
            parser.error(f"Unknown categories: {', '.join(sorted(unknown))}. Valid choices: {', '.join(valid_categories)}")

    random.seed(args.seed)
    output_dir  = Path(args.output_dir)
    prompt_text = load_prompt()
    exclusions  = load_exclusions()
    client      = anthropic.Anthropic(api_key=API_KEY)

    # Build test set
    sources = [s for s in TEST_SOURCES if not args.categories or s["name"] in args.categories]
    test_set = []
    for source in sources:
        for pdf_path in sample_pdfs(source, exclusions, n=2):
            test_set.append({"name": source["name"], "path": pdf_path})

    print(f"Test set — {len(test_set)} papers:")
    for item in test_set:
        print(f"  [{item['name']}] {item['path'].stem}")
    print()

    # Copy selected PDFs so the exact test set is preserved and inspectable
    for item in test_set:
        dest_dir = output_dir / "selected_pdfs" / item["name"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["path"], dest_dir / item["path"].name)

    # Review each paper
    token_rows = []
    for item in test_set:
        name, pdf_path = item["name"], item["path"]
        stem = pdf_path.stem
        print(f"[{name}] {stem}")
        print("  Calling Claude ...", end=" ", flush=True)

        create_kwargs = dict(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": make_content(pdf_path, prompt_text, args.extract_text)}],
        )
        if ENABLE_THINKING:
            create_kwargs["thinking"] = {"type": "adaptive"}
            create_kwargs["extra_body"] = {"output_config": {"effort": THINKING_EFFORT}}
        response = client.messages.create(**create_kwargs)

        text  = next(block.text for block in response.content if block.type == "text")
        usage = response.usage
        n_thinking = sum(1 for b in response.content if b.type == "thinking") if ENABLE_THINKING else 0
        thinking_note = f"  thinking_blocks={n_thinking}" if ENABLE_THINKING else ""
        print(f"done  (in={usage.input_tokens:,}  out={usage.output_tokens:,}){thinking_note}")

        out_dir = output_dir / "reviews" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{stem}.md").write_text(text, encoding="utf-8")
        token_rows.append((name, stem, usage.input_tokens, usage.output_tokens))

    # Summary table
    print("\n" + "=" * 84)
    print("TOKEN SUMMARY")
    print("=" * 84)
    print(f"  {'Category':<20}  {'Paper':<42}  {'Input':>7}  {'Output':>7}")
    print("  " + "-" * 80)
    for cat, stem, inp, out in token_rows:
        print(f"  {cat:<20}  {stem[:42]:<42}  {inp:>7,}  {out:>7,}")
    print("  " + "-" * 80)
    total_in  = sum(r[2] for r in token_rows)
    total_out = sum(r[3] for r in token_rows)
    n         = len(token_rows)
    print(f"  {'TOTAL':<20}  {'':42}  {total_in:>7,}  {total_out:>7,}")
    print(f"  {'AVERAGE':<20}  {'':42}  {total_in // n:>7,}  {total_out // n:>7,}")


if __name__ == "__main__":
    main()
