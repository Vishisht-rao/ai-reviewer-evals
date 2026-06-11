#!/usr/bin/env python3
"""
ReviewerToo API Client

A standalone script for submitting papers to the ReviewerToo API and
collecting results. Designed for the TMLR benchmarking use case.

Dependencies: pip install requests

Usage:
  # Single paper
  python reviewertoo_client.py paper.pdf

  # Multiple papers
  python reviewertoo_client.py paper1.pdf paper2.pdf paper3.pdf

  # All PDFs in a directory
  python reviewertoo_client.py papers/

  # Custom settings
  python reviewertoo_client.py --venue tmlr --output results/ --parallel 5 papers/

Environment:
  REVIEWERTOO_API_URL   Base URL (default: https://reviewertoo.org)
  REVIEWERTOO_API_KEY   API key for authentication
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    print("Error: 'requests' package required. Install with: pip install requests")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional for the standalone client


# Status codes worth retrying: connection-level + Cloudflare/origin transient errors.
# 408 timeout, 425 too early, 429 rate limit, 500 internal, 502 bad gateway,
# 503 unavailable, 504 gateway timeout, 520-524 Cloudflare origin errors.
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}


class ReviewerTooClient:
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["X-API-Key"] = api_key

    def _request_with_retry(
        self, method: str, url: str, *, max_attempts: int = 5, **kwargs
    ) -> requests.Response:
        """Issue a request and retry on transient network/HTTP errors.

        Backs off exponentially (2s, 4s, 8s, 16s) between attempts. Re-raises
        on non-retryable HTTP errors or once max_attempts is exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = self.session.request(method, url, **kwargs)
                if resp.status_code in _RETRYABLE_STATUS and attempt < max_attempts:
                    last_exc = requests.HTTPError(
                        f"{resp.status_code} {resp.reason}", response=resp
                    )
                else:
                    resp.raise_for_status()
                    return resp
            except (requests.ConnectionError, requests.Timeout) as e:
                last_exc = e
            time.sleep(min(2 ** attempt, 30))
        assert last_exc is not None
        raise last_exc

    def list_venues(self) -> dict:
        resp = self._request_with_retry("GET", f"{self.api_url}/api/v1/venues")
        return resp.json()

    def submit_paper(
        self, pdf_path: str, venue: str = "tmlr", smoke: bool = False
    ) -> dict:
        """Submit a single paper for review. Returns task info."""
        params = {"venue": venue}
        if smoke:
            params["smoke"] = "true"
        # Re-open the file on each retry attempt so the upload stream is fresh.
        last_exc: Exception | None = None
        for attempt in range(1, 6):
            with open(pdf_path, "rb") as f:
                try:
                    resp = self.session.post(
                        f"{self.api_url}/api/v1/review",
                        files={"file": (os.path.basename(pdf_path), f, "application/pdf")},
                        params=params,
                    )
                    if resp.status_code in _RETRYABLE_STATUS and attempt < 5:
                        last_exc = requests.HTTPError(
                            f"{resp.status_code} {resp.reason}", response=resp
                        )
                    else:
                        resp.raise_for_status()
                        return resp.json()
                except (requests.ConnectionError, requests.Timeout) as e:
                    last_exc = e
            time.sleep(min(2 ** attempt, 30))
        assert last_exc is not None
        raise last_exc

    def submit_batch(self, pdf_paths: list, venue: str = "tmlr") -> dict:
        """Submit multiple papers at once via the batch endpoint."""
        files = []
        for path in pdf_paths:
            files.append(
                ("files", (os.path.basename(path), open(path, "rb"), "application/pdf"))
            )
        try:
            resp = self.session.post(
                f"{self.api_url}/api/v1/review/batch",
                files=files,
                params={"venue": venue},
            )
            resp.raise_for_status()
            return resp.json()
        finally:
            for _, (_, f, _) in files:
                f.close()

    def get_status(self, task_id: str) -> dict:
        """Get task status and results. Retries on transient 5xx / connection errors."""
        resp = self._request_with_retry(
            "GET", f"{self.api_url}/api/v1/review/{task_id}"
        )
        return resp.json()

    def poll_until_done(
        self, task_id: str, interval: float = 10.0, timeout: float = 3600.0,
        on_poll=None,
    ) -> dict:
        """Poll a task until it completes or fails.

        get_status() already retries transient HTTP/network errors internally.
        This loop only needs to bound how long we wait overall.

        Args:
            on_poll: Optional callback(status_str, elapsed_secs) called each poll cycle.
        """
        start = time.time()
        consecutive_errors = 0
        while True:
            elapsed = time.time() - start
            try:
                status = self.get_status(task_id)
                consecutive_errors = 0
                if status["status"] in ("completed", "failed"):
                    return status
                if on_poll:
                    on_poll(status["status"], elapsed)
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                consecutive_errors += 1
                if on_poll:
                    on_poll(f"transient error (retry {consecutive_errors}/10)", elapsed)
                if consecutive_errors > 10:
                    return {
                        "task_id": task_id,
                        "status": "failed",
                        "error": f"Lost connection to server: {e}",
                    }
            if elapsed > timeout:
                return {
                    "task_id": task_id,
                    "status": "timeout",
                    "error": f"Timed out after {timeout}s",
                }
            time.sleep(interval)

    def download_bundle(self, task_id: str, output_dir: str) -> str | None:
        """Download the full review bundle as a zip file."""
        try:
            resp = self._request_with_retry(
                "GET", f"{self.api_url}/download/{task_id}"
            )
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            raise
        os.makedirs(output_dir, exist_ok=True)
        zip_path = os.path.join(output_dir, f"{task_id}.zip")
        with open(zip_path, "wb") as f:
            f.write(resp.content)
        return zip_path


def collect_pdfs(paths: list) -> list:
    """Resolve paths to a flat list of PDF files."""
    pdfs = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            pdfs.extend(sorted(path.glob("*.pdf")))
        elif path.is_file() and path.suffix.lower() == ".pdf":
            pdfs.append(path)
        else:
            print(f"Warning: skipping '{p}' (not a PDF or directory)")
    return [str(p) for p in pdfs]


def main():
    parser = argparse.ArgumentParser(
        description="ReviewerToo API Client — submit papers for AI review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s paper.pdf
  %(prog)s --venue tmlr papers/
  %(prog)s --parallel 3 --output results/ paper1.pdf paper2.pdf
        """,
    )
    parser.add_argument("inputs", nargs="+", help="PDF file(s) or directory of PDFs")
    parser.add_argument(
        "--api-url",
        default=os.getenv("REVIEWERTOO_API_URL", "https://reviewertoo.org"),
        help="API base URL (default: $REVIEWERTOO_API_URL or https://reviewertoo.org)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("REVIEWERTOO_API_KEY", ""),
        help="API key (default: $REVIEWERTOO_API_KEY)",
    )
    parser.add_argument(
        "--venue",
        default="tmlr",
        choices=["tmlr", "iclr", "icml", "sigmod"],
        help="Target venue (default: tmlr)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="reviewertoo_output",
        help="Output directory for results (default: reviewertoo_output)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=5,
        help="Max concurrent submissions (default: 5)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=15.0,
        help="Seconds between status polls (default: 15)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="Max seconds to wait per paper (default: 3600)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use batch endpoint (single request for all files)",
    )
    parser.add_argument(
        "--no-download", action="store_true", help="Skip downloading review bundles"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test mode: skip LitLLM, run 1 reviewer only (~1-2 min)",
    )
    args = parser.parse_args()

    # Collect PDFs
    pdfs = collect_pdfs(args.inputs)
    if not pdfs:
        print("Error: no PDF files found.")
        sys.exit(1)
    print(f"Found {len(pdfs)} paper(s) to review")

    # Init client
    client = ReviewerTooClient(args.api_url, args.api_key)

    # Verify connection
    try:
        venues = client.list_venues()
        print(
            f"Connected to {args.api_url} — available venues: {list(venues['venues'].keys())}"
        )
    except Exception as e:
        print(f"Error: could not connect to {args.api_url}: {e}")
        sys.exit(1)

    print(f"\nProcessing {len(pdfs)} paper(s) at venue '{args.venue}' "
          f"(up to {args.parallel} in flight)...\n")
    os.makedirs(args.output, exist_ok=True)
    results_summary = []

    # Append-only log of every (filename, task_id, status) the moment we know it.
    # If the script crashes mid-batch, every submitted task_id is recoverable from this file.
    tasks_log_path = os.path.join(args.output, "tasks.jsonl")
    tasks_log_lock = __import__("threading").Lock()

    def log_task(filename, task_id, status, error=None):
        with tasks_log_lock:
            with open(tasks_log_path, "a") as f:
                f.write(json.dumps({
                    "filename": filename,
                    "task_id": task_id,
                    "status": status,
                    "error": error,
                }) + "\n")

    def save_result(task_id, filename, result):
        """Persist a completed task's outputs to disk."""
        paper_stem = Path(filename).stem
        paper_output_dir = os.path.join(args.output, paper_stem)
        os.makedirs(paper_output_dir, exist_ok=True)

        metareview = result.get("metareview")
        if metareview:
            if isinstance(metareview, dict):
                meta_path = os.path.join(paper_output_dir, "metareview.json")
                with open(meta_path, "w") as f:
                    json.dump(metareview, f, indent=2, ensure_ascii=False)
            else:
                meta_path = os.path.join(paper_output_dir, "metareview.md")
                with open(meta_path, "w") as f:
                    f.write(metareview)
            print(f"    -> {filename}: saved metareview")

        if result.get("related_work_summary"):
            rw_path = os.path.join(paper_output_dir, "related_work_summary.md")
            with open(rw_path, "w") as f:
                f.write(result["related_work_summary"])
            print(f"    -> {filename}: saved related work summary")

        reviews = result.get("reviews")
        if reviews:
            reviews_dir = os.path.join(paper_output_dir, "reviews")
            os.makedirs(reviews_dir, exist_ok=True)
            for review in reviews:
                reviewer_id = review.get("reviewer_id", "unknown")
                if "summary_of_contributions" in review:
                    review_path = os.path.join(reviews_dir, f"{reviewer_id}.json")
                    with open(review_path, "w") as f:
                        json.dump(review, f, indent=2, ensure_ascii=False)
                else:
                    review_path = os.path.join(reviews_dir, f"{reviewer_id}.md")
                    with open(review_path, "w") as f:
                        f.write(review.get("raw_review", str(review)))
            print(f"    -> {filename}: saved {len(reviews)} anonymized reviews")

        if not args.no_download:
            try:
                zip_path = client.download_bundle(task_id, paper_output_dir)
                if zip_path:
                    print(f"    -> {filename}: downloaded bundle")
            except Exception as e:
                # Metareview + reviews are already saved via the status endpoint;
                # bundle is a bonus, so don't fail the whole paper if it 403/404s.
                print(f"    -> {filename}: bundle download skipped ({e})")

    def process_one(pdf_path):
        """Submit, poll, and save a single paper end-to-end."""
        filename = os.path.basename(pdf_path)
        try:
            sub = client.submit_paper(pdf_path, venue=args.venue, smoke=args.smoke)
        except Exception as e:
            print(f"  {filename}: submit failed — {e}", flush=True)
            log_task(filename, None, "submit_failed", str(e))
            return {"filename": filename, "task_id": None,
                    "status": "submit_failed", "error": str(e)}

        task_id = sub["task_id"]
        print(f"  Submitted: {filename} -> {task_id[:8]}  (full id logged to tasks.jsonl)", flush=True)
        log_task(filename, task_id, "submitted")

        def _on_poll(status_str, elapsed):
            mins, secs = divmod(int(elapsed), 60)
            print(f"  [{mins:02d}:{secs:02d}] {filename} — {status_str}", flush=True)

        result = client.poll_until_done(
            task_id, interval=args.poll_interval, timeout=args.timeout,
            on_poll=_on_poll,
        )
        status = result["status"]

        if status == "completed":
            print(f"  Done: {filename}", flush=True)
            try:
                save_result(task_id, filename, result)
            except Exception as e:
                print(f"  {filename}: save failed — {e}", flush=True)
        else:
            print(f"  {filename}: {status} — {result.get('error', 'unknown error')}",
                  flush=True)

        log_task(filename, task_id, status, result.get("error"))
        return {"filename": filename, "task_id": task_id,
                "status": status, "error": result.get("error")}

    if args.batch:
        # Original batch path: one submission, then per-task poll workers.
        result = client.submit_batch(pdfs, venue=args.venue)
        print(f"Batch submitted: {result['submitted']} papers\n")
        task_map = {t["task_id"]: t["filename"] for t in result["tasks"]}

        def poll_and_save(task_id, filename):
            def _on_poll(status_str, elapsed):
                mins, secs = divmod(int(elapsed), 60)
                print(f"  [{mins:02d}:{secs:02d}] {filename} — {status_str}", flush=True)
            r = client.poll_until_done(
                task_id, interval=args.poll_interval, timeout=args.timeout,
                on_poll=_on_poll,
            )
            if r["status"] == "completed":
                print(f"  Done: {filename}", flush=True)
                try:
                    save_result(task_id, filename, r)
                except Exception as e:
                    print(f"  {filename}: save failed — {e}", flush=True)
            else:
                print(f"  {filename}: {r['status']} — {r.get('error', 'unknown error')}",
                      flush=True)
            return {"filename": filename, "task_id": task_id,
                    "status": r["status"], "error": r.get("error")}

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = [executor.submit(poll_and_save, tid, fn)
                       for tid, fn in task_map.items()]
            for future in as_completed(futures):
                results_summary.append(future.result())
    else:
        # Stagger submissions: a worker picks up the next PDF only after its
        # previous paper finishes, so at most --parallel papers are in flight
        # on the server at any moment.
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = [executor.submit(process_one, pdf) for pdf in pdfs]
            for future in as_completed(futures):
                results_summary.append(future.result())

    # Write summary
    summary_path = os.path.join(args.output, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    # Print summary
    completed = sum(1 for r in results_summary if r["status"] == "completed")
    failed = sum(1 for r in results_summary if r["status"] == "failed")
    print(f"\n{'=' * 50}")
    print(
        f"Results: {completed} completed, {failed} failed, {len(results_summary) - completed - failed} other"
    )
    print(f"Output directory: {args.output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
