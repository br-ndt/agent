"""Sprite generation + post-processing CLI.

Wraps the project's VisionProvider (Gemini Nano Banana) and layers a PIL
post-pipeline suited for game sprites: chroma-key a solid-color background
to alpha, auto-trim to bbox, optional resize (nearest-neighbor for
pixel-art feel), optional palette quantization.

Subcommands:
  gen      Generate one sprite from a text prompt.
  edit     Generate a variant from a base image + edit prompt.
  split    Cut an existing image (typically a model-returned strip) into
           N tiles; normalize each (chroma-key, trim, uniform pad).
  sheet    Assemble multiple PNG files into a strip or grid sheet.

Run from the repo root (or anywhere — the script re-resolves its parent):
    uv run python tools/sprite_gen.py gen --prompt "..." --output out.png
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from PIL import Image  # noqa: E402

from agent.config import init_dotenv  # noqa: E402


# ── Post-processing ──────────────────────────────────────────


@dataclass
class PostOpts:
    bg_color: tuple[int, int, int] | None
    bg_tolerance: int
    trim: bool
    size: tuple[int, int] | None
    quantize: int | None


def _parse_hex_color(s: str) -> tuple[int, int, int]:
    s = s.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError(f"invalid hex color: {s!r}")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def _parse_size(s: str) -> tuple[int, int]:
    if "x" in s.lower():
        w, h = s.lower().split("x", 1)
        return (int(w), int(h))
    n = int(s)
    return (n, n)


def _chroma_key(img: Image.Image, bg: tuple[int, int, int], tol: int) -> Image.Image:
    """Set alpha=0 for pixels within `tol` (Chebyshev distance) of `bg`."""
    img = img.convert("RGBA")
    px = img.load()
    br, bg_g, bb = bg
    w, h = img.size
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if abs(r - br) <= tol and abs(g - bg_g) <= tol and abs(b - bb) <= tol:
                px[x, y] = (r, g, b, 0)
    return img


def _auto_trim(img: Image.Image) -> Image.Image:
    bbox = img.getbbox()
    return img.crop(bbox) if bbox else img


def _resize(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return img.resize(size, Image.NEAREST)


def _quantize_rgba(img: Image.Image, colors: int) -> Image.Image:
    """Palette-quantize while preserving alpha transparency."""
    rgba = img.convert("RGBA")
    alpha = rgba.split()[-1]
    rgb = rgba.convert("RGB")
    quantized = rgb.quantize(colors=max(2, colors), method=Image.Quantize.MEDIANCUT)
    out = quantized.convert("RGBA")
    out.putalpha(alpha)
    return out


def post_process(data: bytes, opts: PostOpts) -> Image.Image:
    img = Image.open(BytesIO(data)).convert("RGBA")
    if opts.bg_color is not None:
        img = _chroma_key(img, opts.bg_color, opts.bg_tolerance)
    if opts.trim:
        img = _auto_trim(img)
    if opts.size is not None:
        img = _resize(img, opts.size)
    if opts.quantize is not None:
        img = _quantize_rgba(img, opts.quantize)
    return img


# ── Model wrappers ───────────────────────────────────────────


def _vision():
    from agent.providers.vision import VisionProvider

    init_dotenv()
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set (check .env in repo root)")
    return VisionProvider(api_key=key)


async def _generate(prompt: str, aspect_ratio: str, retries: int) -> bytes:
    vp = _vision()
    last: Exception | None = None
    for attempt in range(retries):
        try:
            images = await vp.generate_image(prompt, num_images=1, aspect_ratio=aspect_ratio)
            if images:
                return images[0]
            last = RuntimeError("empty response")
        except Exception as e:
            last = e
        # brief pause between retries
        await asyncio.sleep(1 + attempt)
    raise RuntimeError(f"generation failed after {retries} attempts: {last}")


async def _edit(base: bytes, prompt: str, retries: int) -> bytes:
    vp = _vision()
    last: Exception | None = None
    for attempt in range(retries):
        try:
            images = await vp.edit_image(base, prompt, mime_type="image/png")
            if images:
                return images[0]
            last = RuntimeError("empty response")
        except Exception as e:
            last = e
        await asyncio.sleep(1 + attempt)
    raise RuntimeError(f"edit failed after {retries} attempts: {last}")


# ── Split (model-returned strip → normalized frames) ─────────


def _split_image(
    img: Image.Image,
    frames: int | None,
    grid: tuple[int, int] | None,
) -> list[Image.Image]:
    """Split a composite image into tiles. Either `frames` (horizontal) or `grid` (RxC)."""
    w, h = img.size
    tiles: list[Image.Image] = []
    if grid is not None:
        rows, cols = grid
        tw, th = w // cols, h // rows
        for r in range(rows):
            for c in range(cols):
                box = (c * tw, r * th, (c + 1) * tw, (r + 1) * th)
                tiles.append(img.crop(box))
    else:
        n = frames or 1
        tw = w // n
        for i in range(n):
            box = (i * tw, 0, (i + 1) * tw, h)
            tiles.append(img.crop(box))
    return tiles


def _normalize_frames(frames: list[Image.Image]) -> list[Image.Image]:
    """After per-frame trim, pad each to a common bbox so motion reads cleanly."""
    max_w = max(f.width for f in frames)
    max_h = max(f.height for f in frames)
    out = []
    for f in frames:
        canvas = Image.new("RGBA", (max_w, max_h), (0, 0, 0, 0))
        # center-bottom anchor (stable ground line for walk/idle cycles)
        x = (max_w - f.width) // 2
        y = max_h - f.height
        canvas.paste(f, (x, y), f)
        out.append(canvas)
    return out


# ── Sheet assembly ───────────────────────────────────────────


def _assemble_sheet(
    frames: list[Image.Image],
    grid: tuple[int, int] | None,
    pad: int,
) -> Image.Image:
    if not frames:
        raise ValueError("no frames to assemble")
    w = max(f.width for f in frames)
    h = max(f.height for f in frames)
    if grid is not None:
        rows, cols = grid
    else:
        rows, cols = 1, len(frames)
    sheet_w = cols * w + (cols - 1) * pad
    sheet_h = rows * h + (rows - 1) * pad
    sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        if i >= rows * cols:
            break
        r, c = divmod(i, cols)
        x = c * (w + pad) + (w - f.width) // 2
        y = r * (h + pad) + (h - f.height)
        sheet.paste(f, (x, y), f)
    return sheet


# ── CLI ──────────────────────────────────────────────────────


def _add_post_args(p: argparse.ArgumentParser):
    p.add_argument("--bg-color", default="#FF00FF", help="chroma-key color (default #FF00FF); use 'none' to skip")
    p.add_argument("--bg-tolerance", type=int, default=30, help="per-channel tolerance for chroma-key (0-255)")
    p.add_argument("--no-trim", action="store_true", help="skip auto-trim to bbox")
    p.add_argument("--size", default=None, help="resize after trim, e.g. 64 or 128x96 (nearest-neighbor)")
    p.add_argument("--quantize", type=int, default=None, help="palette quantize to N colors")


def _post_opts_from_args(args) -> PostOpts:
    bg = None if (args.bg_color or "").lower() == "none" else _parse_hex_color(args.bg_color)
    size = _parse_size(args.size) if args.size else None
    return PostOpts(
        bg_color=bg,
        bg_tolerance=max(0, min(255, args.bg_tolerance)),
        trim=not args.no_trim,
        size=size,
        quantize=args.quantize,
    )


def _cmd_gen(args) -> int:
    opts = _post_opts_from_args(args)
    data = asyncio.run(_generate(args.prompt, args.aspect_ratio, args.retries))
    img = post_process(data, opts)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "PNG")
    print(f"wrote {out} ({img.width}x{img.height})")
    return 0


def _cmd_edit(args) -> int:
    opts = _post_opts_from_args(args)
    base = Path(args.input).read_bytes()
    data = asyncio.run(_edit(base, args.prompt, args.retries))
    img = post_process(data, opts)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "PNG")
    print(f"wrote {out} ({img.width}x{img.height})")
    return 0


def _cmd_split(args) -> int:
    opts = _post_opts_from_args(args)
    src = Image.open(args.input).convert("RGBA")
    grid = None
    if args.grid:
        rows, cols = args.grid.lower().split("x", 1)
        grid = (int(rows), int(cols))
    tiles = _split_image(src, args.frames, grid)
    # post-process each tile (chroma-key + trim), then normalize to common bbox
    processed: list[Image.Image] = []
    for t in tiles:
        buf = BytesIO()
        t.save(buf, "PNG")
        processed.append(post_process(buf.getvalue(), opts))
    if args.normalize:
        processed = _normalize_frames(processed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(processed):
        p = out_dir / f"{args.prefix}{i:02d}.png"
        f.save(p, "PNG")
        print(f"wrote {p} ({f.width}x{f.height})")
    return 0


def _cmd_sheet(args) -> int:
    frames = [Image.open(p).convert("RGBA") for p in args.inputs]
    grid = None
    if args.grid:
        rows, cols = args.grid.lower().split("x", 1)
        grid = (int(rows), int(cols))
    sheet = _assemble_sheet(frames, grid, args.pad)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, "PNG")
    print(f"wrote {out} ({sheet.width}x{sheet.height}, {len(frames)} frames)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sprite_gen", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("gen", help="generate a sprite from text")
    p_gen.add_argument("--prompt", required=True)
    p_gen.add_argument("--output", "-o", required=True)
    p_gen.add_argument("--aspect-ratio", default="1:1", help='e.g. "1:1", "16:9"; strips usually want 16:9 or 4:3')
    p_gen.add_argument("--retries", type=int, default=3)
    _add_post_args(p_gen)
    p_gen.set_defaults(func=_cmd_gen)

    p_edit = sub.add_parser("edit", help="edit a base image with a prompt")
    p_edit.add_argument("--input", "-i", required=True)
    p_edit.add_argument("--prompt", required=True)
    p_edit.add_argument("--output", "-o", required=True)
    p_edit.add_argument("--retries", type=int, default=3)
    _add_post_args(p_edit)
    p_edit.set_defaults(func=_cmd_edit)

    p_split = sub.add_parser("split", help="split one image into tiles (post-process each)")
    p_split.add_argument("--input", "-i", required=True)
    p_split.add_argument("--output-dir", "-o", required=True)
    p_split.add_argument("--prefix", default="frame_", help="filename prefix for tiles")
    p_split.add_argument("--frames", type=int, default=None, help="horizontal frame count")
    p_split.add_argument("--grid", default=None, help='grid layout "RxC" (overrides --frames)')
    p_split.add_argument("--normalize", action="store_true", help="pad all tiles to a common bbox (center-bottom anchor)")
    _add_post_args(p_split)
    p_split.set_defaults(func=_cmd_split)

    p_sheet = sub.add_parser("sheet", help="assemble PNGs into a sprite sheet")
    p_sheet.add_argument("inputs", nargs="+", help="PNG files, in order")
    p_sheet.add_argument("--output", "-o", required=True)
    p_sheet.add_argument("--grid", default=None, help='grid layout "RxC" (default: horizontal strip)')
    p_sheet.add_argument("--pad", type=int, default=0, help="pixel padding between frames")
    p_sheet.set_defaults(func=_cmd_sheet)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
