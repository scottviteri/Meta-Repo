# Epstein Files Investigation — Instructions & Coordination Hub

## How This Works

This document coordinates parallel investigation across multiple Claude conversations within a single Claude Project. Each conversation picks up work, logs what it does here, and avoids duplicating effort.

**Before starting work**: Check the "Investigated Documents" list in `FINDINGS.md`.
**While working**: Note your conversation ID and what you're examining.
**After finishing**: Update `FINDINGS.md` with EFTA numbers and findings.

---

## Data Sources

### Available on Internet Archive

| Dataset | Size | Status | Notes |
|---------|------|--------|-------|
| Dataset 9 (partial torrent) | ~46 GB | 1.4% downloaded | WARC from `combined-all-epstein-files` archive |
| Dataset 10 | 82 GB | Not started | Full torrent: `magnet:?xt=urn:btih:d509cc4ca1a415a9ba3b6cb920f67c44aed7fe1f` |
| Dataset 12 | 114 MB | Not started | Torrent: `magnet:?xt=urn:btih:8bc781c7259f4b82406cd2175a1d5e9c3b6bfc90` |
| Datasets 1-8 | Varies | Not started | ZIP files from justice.gov (some removed) |
| Dataset 11 | Unknown | Not started | ZIP removed from DOJ site |

### Downloaded WARCs (Dataset 9)

| File | Size | PDFs | EFTA Range | Keyword Scanned | Deep Read |
|------|------|------|------------|-----------------|-----------|
| `dataset9-00000.warc.gz` (370MB subset) | 370 MB | 2,043 | EFTA00053963–EFTA01260014 | 100% | 25 docs (~1.2%) |
| `dataset9_00000.warc.gz` (full shard) | 660 MB | 3,176 | EFTA00053963–EFTA00480675 | 64% | Same 25 |
| Remaining shards | ~45 GB | ~hundreds of thousands | Unknown | No | No |

### Direct PDF Access (no download needed)

Individual PDFs still accessible at: `https://www.justice.gov/epstein/files/DataSet9/{EFTA_NUMBER}.pdf`
Page listings at: `https://www.justice.gov/epstein/doj-disclosures/data-set-9-files?page={N}`

---

## How to Start a New Investigation Session

```
1. Read this coordination doc
2. Check FINDINGS.md for already-investigated documents
3. Pick an unsummarized document, download, and summarize
4. Update FINDINGS.md with findings
5. Commit and push
```

---

## Extracting PDFs from DOJ Epstein Dataset 9 (Internet Archive WARC)

### Background

The DOJ released Epstein case files at `justice.gov/epstein/files/`. Dataset 9 was broken on the DOJ site (pagination looped, ZIP download corrupted), so an archivist brute-forced all possible PDF filenames (EFTA00033148 through EFTA01262782) and stored the results as WARC files on the Internet Archive.

**Archive URL:** https://archive.org/details/www.justice.gov_epstein_files_DataSet_9_individual_pdf_bruteforce
**Total size:** 92.8 GB across 19 WARC files (~5 GB each), plus CDX index files (~2 MB each)
**Content:** ~1.2 million filenames attempted; actual PDFs are sparse within that range. Most entries are `warc/revisit` (duplicate 404 responses). Real PDFs have MIME type `application/pdf` and HTTP status `200` in the CDX.

### How to extract individual PDFs without downloading the full archive

The technique exploits three properties of the archive format:

1. **CDX index files** list every record's URL, byte offset, and compressed size within the WARC
2. **WARC.GZ files** use concatenated individually-gzipped records (not one big gzip stream), so any single record can be decompressed independently
3. **HTTP Range requests** let you download just the bytes for one record from Archive.org's servers

#### Step 1: Download and parse a CDX index

```bash
# Download a CDX index (~2MB) for one of the 19 WARC files
curl -sL -o index.cdx.gz \
  "https://archive.org/download/www.justice.gov_epstein_files_DataSet_9_individual_pdf_bruteforce/www.justice.gov_epstein_files_DataSet_9-individual-pdfs-bruteforce-00000.warc.os.cdx.gz"

# List only real PDFs (filter out warc/revisit dedup records)
gunzip -c index.cdx.gz | grep -v "warc/revisit" | grep "application/pdf"
```

CDX fields (space-separated): `canonicalized_url timestamp original_url mime_type http_status sha1_digest - - compressed_size byte_offset warc_filename`

The two critical fields for extraction are:
- **byte_offset** (field 10, 0-indexed field 9): where the gzip stream starts in the .warc.gz file
- **compressed_size** (field 9, 0-indexed field 8): how many bytes to fetch

#### Step 2: Range-request a single WARC record

```bash
# Example: EFTA00040230.pdf — size=24516, offset=2799873120 in warc-00000.warc.gz
OFFSET=2799873120
SIZE=24516
END=$((OFFSET + SIZE - 1))

curl -sL -r "${OFFSET}-${END}" -o record.warc.gz \
  "https://archive.org/download/www.justice.gov_epstein_files_DataSet_9_individual_pdf_bruteforce/www.justice.gov_epstein_files_DataSet_9-individual-pdfs-bruteforce-00000.warc.gz"
```

This downloads ~24KB instead of 5GB.

#### Step 3: Decompress and extract the PDF

```bash
gunzip -c record.warc.gz > record.warc
```

Then extract the PDF payload (Python):

```python
data = open('record.warc', 'rb').read()

# WARC record structure: WARC headers \r\n\r\n HTTP response \r\n\r\n body
warc_end = data.find(b'\r\n\r\n')
http_start = warc_end + 4
http_end = data.find(b'\r\n\r\n', http_start)
body = data[http_end + 4:]

# Find PDF start marker (skip any chunked-encoding overhead)
pdf_start = body.find(b'%PDF')
if pdf_start >= 0:
    pdf_data = body[pdf_start:]
    open('output.pdf', 'wb').write(pdf_data)
```

#### Step 4: Read the PDF

```bash
pip install pymupdf --break-system-packages
```

```python
import fitz
doc = fitz.open('output.pdf')
for page in doc:
    print(page.get_text())
```

### Complete extraction script

```python
import subprocess, os

def extract_pdf_from_warc(filename, offset, size, warc_num="00000"):
    """Extract a single PDF from the WARC archive using range requests."""
    end = offset + size - 1
    warc_url = (
        "https://archive.org/download/"
        "www.justice.gov_epstein_files_DataSet_9_individual_pdf_bruteforce/"
        f"www.justice.gov_epstein_files_DataSet_9-individual-pdfs-bruteforce-{warc_num}.warc.gz"
    )

    # Range request
    subprocess.run([
        "curl", "-sL", "-r", f"{offset}-{end}",
        "-o", f"rec_{filename}.warc.gz", warc_url
    ], check=True)

    # Decompress
    subprocess.run(
        f"gunzip -c rec_{filename}.warc.gz > rec_{filename}.warc",
        shell=True, check=True
    )

    # Extract PDF body
    data = open(f"rec_{filename}.warc", "rb").read()
    warc_end = data.find(b"\r\n\r\n")
    http_end = data.find(b"\r\n\r\n", warc_end + 4)
    body = data[http_end + 4:]
    pdf_start = body.find(b"%PDF")
    if pdf_start < 0:
        raise ValueError(f"No PDF found in WARC record for {filename}")
    pdf_data = body[pdf_start:]
    outpath = f"{filename}.pdf"
    open(outpath, "wb").write(pdf_data)

    # Cleanup
    os.remove(f"rec_{filename}.warc.gz")
    os.remove(f"rec_{filename}.warc")
    return outpath


def render_and_read_pdf(pdf_path, dpi=200):
    """Render PDF pages as images and extract text. Returns (image_paths, text_by_page)."""
    import fitz
    doc = fitz.open(pdf_path)
    image_paths = []
    text_by_page = []
    base = os.path.splitext(os.path.basename(pdf_path))[0]

    for i, page in enumerate(doc):
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_path = f"{base}_page_{i+1:03d}.png"
        pix.save(img_path)
        image_paths.append(img_path)
        text_by_page.append(page.get_text())

    return image_paths, text_by_page
```

### WARC file inventory

There are 19 WARC files (00000-00018) and a corresponding CDX index for each. The CDX `warc_filename` field tells you which WARC contains a given record. Adjust the `warc_num` parameter accordingly.

| WARC | CDX index filename |
|------|-------------------|
| `...-00000.warc.gz` (5.0G) | `...-00000.warc.os.cdx.gz` (2.0M) |
| `...-00001.warc.gz` (5.0G) | `...-00001.warc.os.cdx.gz` (2.0M) |
| ... | ... |
| `...-00018.warc.gz` (2.4G) | `...-00018.warc.os.cdx.gz` (1.1M) |

Base download URL for all files:
`https://archive.org/download/www.justice.gov_epstein_files_DataSet_9_individual_pdf_bruteforce/`

### Key concepts

- **WARC (ISO 28500):** Web archive format that wraps HTTP request/response pairs with metadata headers.
- **WARC.GZ:** Each WARC record is individually gzip-compressed, then concatenated. Each record is independently decompressible given its byte offset — this enables random access.
- **CDX:** An index format mapping URLs to byte offsets within a .warc.gz file.
- **warc/revisit:** A deduplication record type. Most brute-forced filenames returned identical 404 pages stored as revisit records.
- **Range requests:** HTTP feature (`curl -r START-END`) to download a byte range without fetching the whole file. Archive.org supports this.

### Processing a bulk WARC (if already downloaded)

```python
from warcio.archiveiterator import ArchiveIterator
import fitz

warc_file = "dataset9-00000.warc.gz"
with open(warc_file, 'rb') as f:
    for record in ArchiveIterator(f):
        if record.rec_type != 'response':
            continue
        url = record.rec_headers.get_header('WARC-Target-URI')
        if not url or '.pdf' not in url.lower():
            continue
        efta = url.split('/')[-1].replace('.pdf', '')
        payload = record.content_stream().read()
        if not payload.startswith(b'%PDF'):
            continue
        doc = fitz.open(stream=payload, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        # ... your analysis here
```
