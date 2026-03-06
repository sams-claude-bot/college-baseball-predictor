#!/usr/bin/env python3
"""Generate small optimized team-logo PNG variants for UI rendering.

Creates:
- web/static/logos/24/<team_id>.png
- web/static/logos/48/<team_id>.png

Source logos are expected at web/static/logos/<team_id>.png
"""

from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
LOGO_DIR = ROOT / "web" / "static" / "logos"
SIZES = (24, 48)


def build_square_variant(src_path: Path, out_path: Path, size: int) -> None:
    with Image.open(src_path) as img:
        img = img.convert("RGBA")
        img.thumbnail((size, size), Image.Resampling.LANCZOS)

        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        x = (size - img.width) // 2
        y = (size - img.height) // 2
        canvas.paste(img, (x, y), img)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path, format="PNG", optimize=True, compress_level=9)


def main() -> int:
    src_files = sorted([p for p in LOGO_DIR.glob("*.png") if p.is_file()])
    if not src_files:
        print(f"No source logos found in {LOGO_DIR}")
        return 1

    created = 0
    for src in src_files:
        for size in SIZES:
            out = LOGO_DIR / str(size) / src.name
            build_square_variant(src, out, size)
            created += 1

    print(f"Generated {created} variants from {len(src_files)} source logos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
