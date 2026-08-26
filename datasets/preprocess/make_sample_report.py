"""Generate a small HTML sample report from datasets/unified/train.jsonl."""

from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "datasets" / "unified" / "train.jsonl"
OUT = REPO / "datasets" / "unified" / "sample_report.html"


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r.get("source", "unknown"), []).append(r)

    # Prefer image-bearing sources for visual review
    quotas = {
        "contextdp": max(15, n // 4),
        "b4e2": max(20, n // 3),
        "hf_synthetic": max(10, n // 5),
        "hf_manual": max(8, n // 8),
    }
    # Normalize quotas to sum ~= n
    total_q = sum(quotas.get(s, 0) for s in by_source)
    if total_q == 0:
        return rng.sample(rows, min(n, len(rows)))
    scale = n / total_q
    picked: list[dict] = []
    used_ids: set[str] = set()
    for source, bucket in by_source.items():
        k = max(1, int(round(quotas.get(source, n // 10) * scale)))
        k = min(k, len(bucket))
        # Prefer rows with images, then with OCR
        ranked = sorted(
            bucket,
            key=lambda r: (bool(r.get("image_path")), bool(r.get("ocr_text"))),
            reverse=True,
        )
        # Shuffle within preference bands
        with_img = [r for r in ranked if r.get("image_path")]
        without = [r for r in ranked if not r.get("image_path")]
        rng.shuffle(with_img)
        rng.shuffle(without)
        ordered = with_img + without
        for r in ordered[:k]:
            if r["id"] not in used_ids:
                picked.append(r)
                used_ids.add(r["id"])

    if len(picked) < n:
        rest = [r for r in rows if r["id"] not in used_ids]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])

    rng.shuffle(picked)
    return picked[:n]


def img_src(image_path: str | None, report_dir: Path) -> str | None:
    if not image_path:
        return None
    abs_path = (REPO / image_path).resolve()
    if not abs_path.exists():
        return None
    try:
        rel = abs_path.relative_to(report_dir)
        return rel.as_posix()
    except ValueError:
        return abs_path.as_uri()


def render(rows: list[dict], out_path: Path) -> None:
    report_dir = out_path.parent
    cards: list[str] = []
    for i, r in enumerate(rows, 1):
        src = img_src(r.get("image_path"), report_dir)
        ocr = (r.get("ocr_text") or r.get("text") or "").strip()
        if len(ocr) > 1200:
            ocr = ocr[:1200] + "…"
        ocr_html = html.escape(ocr) if ocr else "<em class='muted'>No OCR / text available</em>"
        if src:
            media = f'<img src="{html.escape(src)}" alt="sample {i}" loading="lazy" />'
        else:
            media = '<div class="no-img">Text-only sample<br/>(no screenshot)</div>'

        labels = (
            f"<div><strong>Binary:</strong> {html.escape(str(r.get('label_binary')))}</div>"
            f"<div><strong>Fine:</strong> {html.escape(str(r.get('label_fine')))}</div>"
            f"<div><strong>ConsentShield:</strong> {html.escape(str(r.get('label_consentshield')))}</div>"
            f"<div><strong>Source labels:</strong> {html.escape(', '.join(r.get('source_labels') or []) or '—')}</div>"
        )
        cards.append(
            f"""
<article class="card">
  <header>
    <span class="idx">#{i}</span>
    <code>{html.escape(r.get('id', ''))}</code>
    <span class="chip">{html.escape(str(r.get('source', '')))}</span>
  </header>
  <div class="grid">
    <div class="media">{media}</div>
    <div class="meta">
      <h3>Labels</h3>
      {labels}
      <h3>OCR / text</h3>
      <pre>{ocr_html}</pre>
      <p class="path muted">{html.escape(str(r.get('image_path') or '—'))}</p>
    </div>
  </div>
</article>
"""
        )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ConsentShield Unified Train Sample Report</title>
  <style>
    :root {{
      --bg: #f3f5f4;
      --card: #fff;
      --ink: #14201b;
      --muted: #5c6b64;
      --border: #d5ddd8;
      --accent: #1b5e45;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: linear-gradient(180deg, #e8efe9, var(--bg));
      color: var(--ink);
      line-height: 1.45;
    }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 1.5rem 1rem 3rem; }}
    h1 {{ margin: 0 0 0.35rem; font-size: 1.6rem; }}
    .sub {{ color: var(--muted); margin-bottom: 1.25rem; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 0.9rem 1rem 1rem;
      margin-bottom: 1rem;
      box-shadow: 0 8px 24px rgba(20,32,27,0.06);
    }}
    header {{
      display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center;
      margin-bottom: 0.75rem;
    }}
    .idx {{ font-weight: 700; color: var(--accent); }}
    code {{ font-size: 0.78rem; color: var(--muted); }}
    .chip {{
      background: #e7f2ec; color: var(--accent);
      border-radius: 999px; padding: 0.15rem 0.55rem; font-size: 0.75rem; font-weight: 650;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(220px, 42%) 1fr;
      gap: 1rem;
    }}
    @media (max-width: 800px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    .media img {{
      width: 100%; max-height: 360px; object-fit: contain;
      background: #0b0f0d; border-radius: 10px; border: 1px solid var(--border);
    }}
    .no-img {{
      min-height: 160px; display: grid; place-items: center; text-align: center;
      background: #eef2f0; border-radius: 10px; color: var(--muted); border: 1px dashed var(--border);
    }}
    .meta h3 {{ margin: 0.2rem 0 0.35rem; font-size: 0.92rem; }}
    .meta pre {{
      white-space: pre-wrap; word-break: break-word;
      background: #f7faf8; border: 1px solid var(--border);
      border-radius: 10px; padding: 0.65rem 0.75rem; font-size: 0.78rem;
      max-height: 240px; overflow: auto; margin: 0;
    }}
    .muted {{ color: var(--muted); font-size: 0.78rem; }}
    .path {{ margin: 0.5rem 0 0; word-break: break-all; }}
  </style>
</head>
<body>
  <main>
    <h1>ConsentShield — Train sample report</h1>
    <p class="sub">{len(rows)} random samples from <code>datasets/unified/train.jsonl</code>
      (image · OCR/text · labels). Open this file locally so screenshot paths resolve.</p>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    out_path.write_text(doc, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    rows = load_rows(TRAIN)
    n = max(50, min(100, args.n))
    sample = stratified_sample(rows, n=n, seed=args.seed)
    render(sample, args.out)
    print(f"Wrote {len(sample)} samples -> {args.out}")


if __name__ == "__main__":
    main()
