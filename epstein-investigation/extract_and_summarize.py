#!/usr/bin/env python3
"""
Epstein Dataset 9 — PDF extraction and summarization tool.

Usage:
    # Extract a single PDF by EFTA number (uses CDX index + range request)
    python extract_and_summarize.py extract EFTA00100000

    # Build local CDX catalog for a WARC shard
    python extract_and_summarize.py index 00000

    # List all real PDFs in a CDX index
    python extract_and_summarize.py list-pdfs 00000

    # Extract and read text from a PDF
    python extract_and_summarize.py read EFTA00100000

    # Batch extract N random unread PDFs from a shard
    python extract_and_summarize.py batch 00000 --count 10

Requires: pymupdf (pip install pymupdf)
"""

import argparse
import csv
import gzip
import os
import random
import subprocess
import sys

ARCHIVE_BASE = (
    "https://archive.org/download/"
    "www.justice.gov_epstein_files_DataSet_9_individual_pdf_bruteforce"
)
DOJ_DIRECT = "https://www.justice.gov/epstein/files/DataSet9"
WORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")
CDX_DIR = os.path.join(WORK_DIR, "cdx")
PDF_DIR = os.path.join(WORK_DIR, "pdfs")


def ensure_dirs():
    os.makedirs(CDX_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)


def cdx_filename(shard: str) -> str:
    return f"www.justice.gov_epstein_files_DataSet_9-individual-pdfs-bruteforce-{shard}.warc.os.cdx.gz"


def warc_filename(shard: str) -> str:
    return f"www.justice.gov_epstein_files_DataSet_9-individual-pdfs-bruteforce-{shard}.warc.gz"


def download_cdx(shard: str) -> str:
    """Download CDX index for a shard if not already present."""
    ensure_dirs()
    local_path = os.path.join(CDX_DIR, f"index-{shard}.cdx.gz")
    if os.path.exists(local_path):
        print(f"CDX index already exists: {local_path}")
        return local_path

    url = f"{ARCHIVE_BASE}/{cdx_filename(shard)}"
    print(f"Downloading CDX index for shard {shard}...")
    subprocess.run(
        ["curl", "-sL", "--retry", "3", "-o", local_path, url],
        check=True,
    )
    print(f"Saved to {local_path}")
    return local_path


def parse_cdx(cdx_path: str) -> list[dict]:
    """Parse CDX index, returning only real PDF entries (not warc/revisit)."""
    entries = []
    with gzip.open(cdx_path, "rt", errors="replace") as f:
        for line in f:
            if "warc/revisit" in line:
                continue
            if "application/pdf" not in line:
                continue
            parts = line.strip().split(" ")
            if len(parts) < 11:
                continue
            url = parts[2]
            efta = url.split("/")[-1].replace(".pdf", "")
            entries.append(
                {
                    "url": url,
                    "efta": efta,
                    "mime": parts[3],
                    "status": parts[4],
                    "digest": parts[5],
                    "compressed_size": int(parts[8]),
                    "offset": int(parts[9]),
                    "warc_file": parts[10],
                }
            )
    return entries


def extract_pdf_from_warc(efta: str, offset: int, size: int, shard: str = "00000") -> str:
    """Extract a single PDF from the WARC archive using HTTP range requests."""
    ensure_dirs()
    outpath = os.path.join(PDF_DIR, f"{efta}.pdf")
    if os.path.exists(outpath):
        print(f"PDF already extracted: {outpath}")
        return outpath

    end = offset + size - 1
    warc_url = f"{ARCHIVE_BASE}/{warc_filename(shard)}"
    rec_gz = os.path.join(WORK_DIR, f"rec_{efta}.warc.gz")
    rec_raw = os.path.join(WORK_DIR, f"rec_{efta}.warc")

    print(f"Range-requesting {efta} ({size} bytes at offset {offset})...")
    subprocess.run(
        ["curl", "-sL", "-r", f"{offset}-{end}", "-o", rec_gz, warc_url],
        check=True,
    )

    subprocess.run(f"gunzip -c '{rec_gz}' > '{rec_raw}'", shell=True, check=True)

    data = open(rec_raw, "rb").read()
    warc_end = data.find(b"\r\n\r\n")
    http_end = data.find(b"\r\n\r\n", warc_end + 4)
    body = data[http_end + 4 :]
    pdf_start = body.find(b"%PDF")
    if pdf_start < 0:
        os.remove(rec_gz)
        os.remove(rec_raw)
        raise ValueError(f"No PDF found in WARC record for {efta}")

    pdf_data = body[pdf_start:]
    with open(outpath, "wb") as f:
        f.write(pdf_data)

    os.remove(rec_gz)
    os.remove(rec_raw)
    print(f"Extracted {efta}.pdf ({len(pdf_data)} bytes)")
    return outpath


def extract_pdf_direct(efta: str) -> str:
    """Download PDF directly from justice.gov (if still available)."""
    ensure_dirs()
    outpath = os.path.join(PDF_DIR, f"{efta}.pdf")
    if os.path.exists(outpath):
        print(f"PDF already exists: {outpath}")
        return outpath

    url = f"{DOJ_DIRECT}/{efta}.pdf"
    print(f"Downloading {efta} directly from DOJ...")
    result = subprocess.run(
        ["curl", "-sL", "--retry", "3", "-o", outpath, "-w", "%{http_code}", url],
        capture_output=True,
        text=True,
    )
    status = result.stdout.strip()
    if status != "200":
        if os.path.exists(outpath):
            os.remove(outpath)
        raise ValueError(f"Failed to download {efta}: HTTP {status}")

    print(f"Downloaded {efta}.pdf")
    return outpath


def read_pdf_text(pdf_path: str) -> list[str]:
    """Extract text from each page of a PDF. Returns list of page texts."""
    import fitz

    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return pages


def cmd_index(args):
    cdx_path = download_cdx(args.shard)
    entries = parse_cdx(cdx_path)
    catalog_path = os.path.join(CDX_DIR, f"catalog-{args.shard}.csv")
    with open(catalog_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["efta", "offset", "compressed_size", "warc_file", "url"])
        writer.writeheader()
        for e in entries:
            writer.writerow(
                {
                    "efta": e["efta"],
                    "offset": e["offset"],
                    "compressed_size": e["compressed_size"],
                    "warc_file": e["warc_file"],
                    "url": e["url"],
                }
            )
    print(f"Found {len(entries)} real PDFs in shard {args.shard}")
    print(f"Catalog saved to {catalog_path}")


def cmd_list_pdfs(args):
    cdx_path = download_cdx(args.shard)
    entries = parse_cdx(cdx_path)
    for e in entries:
        print(f"{e['efta']}  size={e['compressed_size']}  offset={e['offset']}")
    print(f"\nTotal: {len(entries)} PDFs")


def cmd_extract(args):
    efta = args.efta.upper()
    if not efta.startswith("EFTA"):
        efta = f"EFTA{efta}"

    if args.direct:
        path = extract_pdf_direct(efta)
    else:
        cdx_path = download_cdx(args.shard)
        entries = parse_cdx(cdx_path)
        match = [e for e in entries if e["efta"] == efta]
        if not match:
            print(f"EFTA {efta} not found in shard {args.shard} CDX index.")
            print("Try --direct flag to download from justice.gov, or a different --shard.")
            sys.exit(1)
        entry = match[0]
        shard = entry["warc_file"].split("-")[-1].replace(".warc.gz", "")
        path = extract_pdf_from_warc(efta, entry["offset"], entry["compressed_size"], shard)

    print(f"PDF at: {path}")


def cmd_read(args):
    efta = args.efta.upper()
    if not efta.startswith("EFTA"):
        efta = f"EFTA{efta}"

    pdf_path = os.path.join(PDF_DIR, f"{efta}.pdf")
    if not os.path.exists(pdf_path):
        print(f"PDF not found locally. Extracting first...")
        if args.direct:
            pdf_path = extract_pdf_direct(efta)
        else:
            cdx_path = download_cdx(args.shard)
            entries = parse_cdx(cdx_path)
            match = [e for e in entries if e["efta"] == efta]
            if not match:
                print(f"Trying direct download from DOJ...")
                pdf_path = extract_pdf_direct(efta)
            else:
                entry = match[0]
                shard = entry["warc_file"].split("-")[-1].replace(".warc.gz", "")
                pdf_path = extract_pdf_from_warc(efta, entry["offset"], entry["compressed_size"], shard)

    pages = read_pdf_text(pdf_path)
    print(f"\n{'='*60}")
    print(f"EFTA: {efta} — {len(pages)} pages")
    print(f"{'='*60}\n")
    for i, text in enumerate(pages):
        print(f"--- Page {i+1} ---")
        print(text.strip() if text.strip() else "[No extractable text — may be scanned image]")
        print()


def cmd_batch(args):
    cdx_path = download_cdx(args.shard)
    entries = parse_cdx(cdx_path)

    # Load already-investigated EFTAs
    findings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINDINGS.md")
    investigated = set()
    if os.path.exists(findings_path):
        with open(findings_path) as f:
            for line in f:
                if line.startswith("| EFTA"):
                    parts = line.split("|")
                    if len(parts) > 1:
                        efta = parts[1].strip()
                        if efta.startswith("EFTA"):
                            investigated.add(efta)

    # Filter to unread
    unread = [e for e in entries if e["efta"] not in investigated]
    print(f"Total PDFs in shard: {len(entries)}")
    print(f"Already investigated: {len(investigated)}")
    print(f"Remaining unread: {len(unread)}")

    if not unread:
        print("All PDFs in this shard have been investigated!")
        return

    # Sample
    count = min(args.count, len(unread))
    sample = random.sample(unread, count)

    for entry in sample:
        shard = entry["warc_file"].split("-")[-1].replace(".warc.gz", "")
        try:
            pdf_path = extract_pdf_from_warc(
                entry["efta"], entry["offset"], entry["compressed_size"], shard
            )
            pages = read_pdf_text(pdf_path)
            total_text = sum(len(p) for p in pages)
            print(f"\n{entry['efta']}: {len(pages)} pages, {total_text} chars of text")
            # Print first 500 chars as preview
            preview = " ".join(p.strip() for p in pages)[:500]
            print(f"  Preview: {preview}...")
        except Exception as e:
            print(f"  ERROR extracting {entry['efta']}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Epstein Dataset 9 investigation tool")
    sub = parser.add_subparsers(dest="command")

    p_index = sub.add_parser("index", help="Download and catalog a CDX index")
    p_index.add_argument("shard", help="Shard number (e.g., 00000)")

    p_list = sub.add_parser("list-pdfs", help="List all PDFs in a shard")
    p_list.add_argument("shard", help="Shard number")

    p_extract = sub.add_parser("extract", help="Extract a single PDF")
    p_extract.add_argument("efta", help="EFTA number (e.g., EFTA00100000)")
    p_extract.add_argument("--shard", default="00000", help="Shard to search (default: 00000)")
    p_extract.add_argument("--direct", action="store_true", help="Download directly from DOJ")

    p_read = sub.add_parser("read", help="Extract and read a PDF")
    p_read.add_argument("efta", help="EFTA number")
    p_read.add_argument("--shard", default="00000")
    p_read.add_argument("--direct", action="store_true")

    p_batch = sub.add_parser("batch", help="Batch extract random unread PDFs")
    p_batch.add_argument("shard", help="Shard number")
    p_batch.add_argument("--count", type=int, default=10, help="Number to extract")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    {"index": cmd_index, "list-pdfs": cmd_list_pdfs, "extract": cmd_extract, "read": cmd_read, "batch": cmd_batch}[args.command](args)


if __name__ == "__main__":
    main()
