"""HTML fragment builders for the HTMX frontend."""
from __future__ import annotations

import html

import psutil

from store import state


def memory_stats() -> str:
    n = len(state.store._docs)
    dim = state.store._index.dim
    bit_width = state.store._index.bit_width
    if n == 0 or dim is None:
        vec_line = f"Vectors: empty ({bit_width}-bit)"
    else:
        q_bytes = n * dim * bit_width / 8
        fp32_bytes = n * dim * 4
        ratio = fp32_bytes / q_bytes

        def _fmt(b: float) -> str:
            return f"{b / (1024 * 1024):.1f} MB" if b >= 1024 * 1024 else f"{b / 1024:.1f} KB"

        vec_line = f"Vectors: {bit_width}-bit · {_fmt(q_bytes)} · {_fmt(fp32_bytes)} FP32 ({ratio:.0f}x)"
    proc_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    return f'{vec_line} · Process: {proc_mb:.0f} MB'


def source_filter_html() -> str:
    sources = sorted({meta.get("source", "?") for _, (_, meta) in state.store._docs.items()})
    options = "".join(
        f'<option value="{html.escape(s, quote=True)}">{html.escape(s)}</option>'
        for s in sources
    )
    return (
        f'<select id="source-filter" multiple size="4" hx-swap-oob="true"'
        f' style="margin-bottom:0.5rem;font-size:0.82rem">{options}</select>'
    )


def doc_list_html(authenticated: bool = True) -> str:
    groups: dict[str, list[tuple[str, str, dict]]] = {}
    for sid, (text, meta) in state.store._docs.items():
        groups.setdefault(meta.get("source", "?"), []).append((sid, text, meta))

    sections = ""
    for source, chunks in groups.items():
        src_escaped = html.escape(source)
        src_url = html.escape(source, quote=True)
        delete_chunk_btn = (
            f'<button style="padding:0 0.3rem;font-size:0.72rem" class="secondary outline"'
            f' hx-delete="/documents/{{}}"'
            f' hx-target="#doc-list" hx-swap="outerHTML"'
            f' hx-confirm="Delete this chunk?">×</button>'
        ) if authenticated else ""
        rows = "".join(
            f'<li style="display:flex;align-items:baseline;gap:0.5rem">'
            f'<em>#{meta.get("chunk", 0) + 1}</em> '
            f'<span style="flex:1">{html.escape(text[:80])}{"…" if len(text) > 80 else ""}</span>'
            + (delete_chunk_btn.format(sid) if authenticated else "")
            + f'</li>'
            for sid, text, meta in chunks
        )
        delete_source_btn = (
            f'<button style="padding:0 0.4rem;font-size:0.72rem;margin-left:auto" class="secondary outline"'
            f' hx-delete="/sources/{src_url}"'
            f' hx-target="#doc-list" hx-swap="outerHTML"'
            f' hx-confirm="Delete all chunks from {src_escaped}?">delete source</button>'
        ) if authenticated else ""
        sections += (
            f'<details style="margin-bottom:0.5rem">'
            f'<summary style="display:flex;align-items:center;gap:0.5rem;cursor:pointer">'
            f'<strong>{src_escaped}</strong>'
            f'<small style="color:var(--pico-muted-color)">{len(chunks)} chunk{"s" if len(chunks) != 1 else ""}</small>'
            + delete_source_btn
            + f'</summary>'
            f'<ul class="doc-list">{rows}</ul>'
            f'</details>'
        )

    total = len(state.store._docs)
    summary = f'{total} chunk{"s" if total != 1 else ""} · {len(groups)} source{"s" if len(groups) != 1 else ""}'
    return (
        f'<div id="doc-list">'
        f'<details open>'
        f'<summary style="font-size:0.85rem;cursor:pointer">{summary}</summary>'
        f'{sections}'
        f'</details>'
        f'<p style="font-size:0.75rem;color:var(--pico-muted-color);margin:0.4rem 0 0">'
        f'{memory_stats()} · chunk_size={state.chunk_size} overlap={state.chunk_overlap}'
        f'{" · contextual=on" if state.contextual else ""}</p>'
        f'</div>'
        f'{source_filter_html()}'
    )
