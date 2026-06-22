"""Text extraction (plain text + scanned PDF OCR) and contextual enrichment."""
from __future__ import annotations

import asyncio
import base64
import io

import anthropic as _sdk
import pypdf

from store import OCR_DIR

_ENRICHMENT_CONCURRENCY = 5
_MAX_SOURCE_CHARS = 50_000  # ~12k tokens — fits in prompt-cache block, avoids overflow


async def enrich_chunks(chunks: list[str], source_text: str) -> list[str]:
    """Prepend a one-sentence context to each chunk (Anthropic Contextual Retrieval).

    Caps concurrent Claude calls via a semaphore and uses prompt caching on the
    source document so only the first call per batch pays full token cost.
    """
    client = _sdk.AsyncAnthropic()
    sem = asyncio.Semaphore(_ENRICHMENT_CONCURRENCY)
    doc_text = source_text[:_MAX_SOURCE_CHARS]
    if len(source_text) > _MAX_SOURCE_CHARS:
        doc_text += "\n\n[document truncated for context]"

    async def _one(chunk: str) -> str:
        async with sem:
            msg = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": f"<document>\n{doc_text}\n</document>",
                     "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": (
                        f"Here is the chunk we want to situate within the whole document:\n"
                        f"<chunk>\n{chunk}\n</chunk>\n\n"
                        "Give a short succinct context to situate this chunk within the overall document "
                        "for the purposes of improving search retrieval. Answer only with the succinct context and nothing else."
                    )},
                ]}],
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            )
            return f"{msg.content[0].text.strip()}\n\n{chunk}"

    return list(await asyncio.gather(*[_one(c) for c in chunks]))


def _extract_with_claude(pdf_bytes: bytes) -> str:
    """OCR a scanned PDF via Claude's native document block."""
    client = _sdk.Anthropic()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=8192,
        messages=[{"role": "user", "content": [
            {"type": "document", "source": {
                "type": "base64", "media_type": "application/pdf",
                "data": base64.standard_b64encode(pdf_bytes).decode(),
            }},
            {"type": "text", "text": "Extract all text from this document, preserving paragraph structure. Output only the extracted text."},
        ]}],
    )
    return msg.content[0].text


def extract_text(filename: str, data: bytes) -> str:
    """Return plain text for a file. Scanned PDFs fall back to Claude OCR with caching."""
    if filename.lower().endswith(".pdf"):
        reader = pypdf.PdfReader(io.BytesIO(data))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        if len(text.strip()) < 100 * max(len(reader.pages), 1):
            ocr_path = OCR_DIR / (filename + ".txt")
            if ocr_path.exists():
                print(f"Using cached OCR for {filename}")
                return ocr_path.read_text()
            print(f"Sparse pypdf output ({len(text.strip())} chars, {len(reader.pages)} pages) — using Claude OCR")
            text = _extract_with_claude(data)
            ocr_path.write_text(text)
            print(f"OCR cached to {ocr_path}")
        return text
    return data.decode(errors="replace")
