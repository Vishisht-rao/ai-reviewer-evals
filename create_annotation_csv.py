"""
Creates annotation CSV files for FLAWS test case categories.

For each agent found in each FLAWS category (gemini / openai), writes TWO CSVs to
  FLAWS_test_cases/{category}/_annotation_csv/{agent_name}_claims_yes.csv
  FLAWS_test_cases/{category}/_annotation_csv/{agent_name}_claims_no.csv

Columns:
  paper_id        — arxiv ID (e.g. 2501.06246v3)
  ground_truth    — full text of ground_truth_errors/{paper_id}.txt
  claims_review   — everything between the claims question and the audience question
                    in the agent's review (includes Yes/No + full explanation)
  annotation      — empty; fill in manually

Usage:
    python create_annotation_csv.py
"""

import csv
from pathlib import Path

BASE_DIR   = Path(__file__).parent
FLAWS_DIR  = BASE_DIR / "FLAWS_test_cases"
CATEGORIES = ["gemini", "openai"]

AGENT_NAMES = {
    "output":                        "CSPaper",
    "output_reviewertoo_metareview": "ReviewerToo",
    "output_claude":                 "Opus_4.7_v2",
    "output_claude_v3":              "Opus_4.7_v3",
}

CLAIMS_Q  = "are the claims made in the submission supported by accurate"
AUDIENCE_Q = "would at least some individuals in tmlr"


def extract_claims_section(path: Path) -> tuple[str, str]:
    """Return (claims_answer, claims_text) where claims_text is everything between
    the claims question heading and the audience question heading."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # Strip YAML frontmatter
    start = 0
    if lines and lines[0].strip() == "---":
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                start = i + 1
                break

    collecting = False
    answer = "MISSING"
    result = []
    for line in lines[start:]:
        stripped = line.strip()
        lower = stripped.lower()
        if CLAIMS_Q in lower:
            collecting = True
            continue
        if collecting:
            if AUDIENCE_Q in lower:
                break
            # Capture first Yes/No as the answer
            if answer == "MISSING" and stripped.capitalize() in ("Yes", "No"):
                answer = stripped.capitalize()
            result.append(line)

    return answer, "\n".join(result).strip()


def paper_id_from_stem(stem: str) -> str:
    """Strip __agent_id suffix if present, returning just the paper ID."""
    return stem.rsplit("__", 1)[0] if "__" in stem else stem


def process_category(category: str) -> None:
    cat_dir   = FLAWS_DIR / category
    gt_dir    = cat_dir / "ground_truth_errors"
    out_root  = cat_dir / "_annotation_csv"
    out_root.mkdir(exist_ok=True)

    # Build ground truth lookup: paper_id -> text
    gt_texts = {}
    if gt_dir.exists():
        for gt_file in gt_dir.glob("*.txt"):
            gt_texts[gt_file.stem] = gt_file.read_text(encoding="utf-8").strip()

    # Discover agents
    for output_dir in sorted(cat_dir.glob("output*/")):
        if not output_dir.is_dir():
            continue
        agent_name = AGENT_NAMES.get(output_dir.name, output_dir.name)

        # Collect review files (.md or .txt)
        reviews = {}
        for f in output_dir.glob("*"):
            if f.suffix not in (".md", ".txt"):
                continue
            pid = paper_id_from_stem(f.stem)
            reviews[pid] = f

        if not reviews:
            print(f"  [{category}/{output_dir.name}] no review files found, skipping")
            continue

        yes_rows, no_rows = [], []
        for pid in sorted(reviews):
            gt_text = gt_texts.get(pid, "")
            answer, claims_text = extract_claims_section(reviews[pid])
            row = {
                "paper_id":      pid,
                "ground_truth":  gt_text,
                "claims_review": claims_text,
                "annotation":    "",
            }
            if answer == "Yes":
                yes_rows.append(row)
            else:
                no_rows.append(row)

        fieldnames = ["paper_id", "ground_truth", "claims_review", "annotation"]
        for suffix, rows in [("claims_yes", yes_rows), ("claims_no", no_rows)]:
            csv_path = out_root / f"{agent_name}_{suffix}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"  [{category}] {agent_name}: wrote {len(rows):3d} rows -> {csv_path.relative_to(BASE_DIR)}")


def main() -> None:
    for cat in CATEGORIES:
        print(f"Processing {cat}")
        process_category(cat)


if __name__ == "__main__":
    main()
