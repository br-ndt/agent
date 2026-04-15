---
title: Sprite Generation for Web Games
tags:
  - sprites
  - game-dev
  - image-gen
  - nano-banana
  - pil
agents:
  - sprite_maker
updated_at: 1776000000
---

Game-ready 2D sprites with Gemini Nano Banana + `tools/sprite_gen.py`.
The model is general-purpose, not sprite-specialized — prompt discipline
does 80% of the work, PIL post-processing handles the rest.

## The pipeline

```
text prompt ──► Nano Banana ──► raw PNG on magenta bg
                                        │
                                        ▼
                             chroma-key → alpha
                                        │
                                        ▼
                                auto-trim to bbox
                                        │
                                        ▼
                              (optional) resize (nearest)
                                        │
                                        ▼
                              (optional) palette quantize
                                        │
                                        ▼
                              game-ready RGBA PNG
```

All of this is driven by `uv run python tools/sprite_gen.py <cmd> ...`.

## Prompting conventions (non-negotiable)

Every generation prompt must include:

1. **Background**: `"on a solid pure magenta (#FF00FF) background, no gradient, no shadow, no ground plane, no ambient scenery"`.
2. **Isolation**: `"single character centered in frame, no other objects"`.
3. **Pose anchor**: specify exactly one pose. For character sheets: `"frontal T-pose, arms out, full body visible head-to-toe"`.
4. **Lighting**: `"flat neutral lighting, no cast shadow"` — shadows confuse chroma-key.
5. **Style lock**: be concrete. `"16-bit SNES pixel art, limited 16-color palette, crisp pixel edges, no anti-aliasing"` or `"flat vector cartoon, bold black outlines, cel-shaded"`. Vague style = random style.
6. **Framing**: `"character fills ~70% of frame height, centered, symmetric margins"`.

### Style presets worth memorizing

- **Pixel art (NES-ish)**: `"8-bit NES pixel art, 3-color palette, stark black outlines, no dithering, no anti-aliasing"` + `--quantize 8 --size 32`.
- **Pixel art (SNES-ish)**: `"16-bit SNES RPG sprite, 16-color palette, subtle shading, crisp edges, no anti-aliasing"` + `--quantize 16 --size 64`.
- **Vector/cartoon**: `"flat vector illustration, bold 2px black outline, cel-shaded with 2 tones, saturated colors"` — usually no quantize needed.
- **Hand-painted**: `"painted 2D game sprite, soft brush strokes, minimal outline, warm palette"` — skip quantize, larger size.

## Character + animation workflow

**Rule: never generate animation frames in separate calls.** The model has
no memory of prior generations; the silhouettes, proportions, and colors
will drift. Instead:

### Walk/idle cycle (2–8 frames)

1. One `gen` call for the whole strip:
   ```
   "4-frame walk cycle of <character>, side view, frames arranged left-to-right
    in a horizontal strip, evenly spaced, each frame shows: (1) contact pose,
    (2) recoil, (3) passing, (4) high point. Same character, identical
    proportions across all frames. On solid pure magenta (#FF00FF) background,
    flat lighting, no shadows, no ground line."
   ```
   Use `--aspect-ratio 16:9` or wider.
2. `split --frames 4 --normalize` to cut and pad to a common bbox
   (center-bottom anchor keeps the feet planted).
3. Inspect each frame; regenerate if one is broken.
4. `sheet` to assemble a final strip if the engine wants one file.

### Multi-directional character (N/S/E/W sprites)

Same pattern: one generation with `"four views of the same character in a
2x2 grid: top-left facing up, top-right facing right, bottom-left facing
down, bottom-right facing left. Identical proportions and palette across
all four."`, then `split --grid 2x2 --normalize`.

### Variants / reskins

Use `edit` on a known-good base sprite:
```bash
uv run python tools/sprite_gen.py edit \
  -i base_knight.png --prompt "same sprite but with a red cape and gold trim" \
  -o red_knight.png
```
`edit` is far more consistent than a fresh `gen` for "same character but X."

## Post-processing knobs

- **`--bg-tolerance`** (default 30): lower values preserve colors that happen to
  be close to magenta (e.g. a purple cloak); higher values kill magenta fringing
  from JPEG-like compression. If you see magenta edges, bump to 40–50.
- **`--size`**: always applied *after* trim, so the character fills the target
  resolution. For pixel art, pair with `--quantize` for clean edges.
- **`--quantize N`**: reduces to N palette colors, then re-applies alpha.
  Good for retro aesthetics and for hiding minor rendering artifacts. Typical
  values: 8 (NES), 16 (SNES), 32 (Genesis-ish), omit for modern/vector.
- **`--no-trim`**: only use when preserving a specific canvas size matters
  (e.g. fixed-tile backgrounds).

## Common failure modes

| Symptom                              | Fix                                                                        |
|--------------------------------------|----------------------------------------------------------------------------|
| Background isn't solid magenta       | Tighten prompt: `"solid pure #FF00FF"`, `"no gradient"`. Regenerate.       |
| Magenta halo / fringing on edges     | Increase `--bg-tolerance` to 40–50.                                        |
| Characters drift across frames       | Generate as one strip, not N separate calls. Re-read section above.        |
| Blurry when upscaled in engine       | Set `--size` to the target resolution. Nearest-neighbor is locked in.     |
| Cast shadow becomes a dark smudge    | Prompt must say `"no shadow, no ground plane, flat lighting"`.             |
| Anti-aliased fringe in pixel art     | Prompt `"no anti-aliasing, crisp hard edges"` + `--quantize 8`.            |
| Model refuses sensitive content      | Don't fight it — rephrase or request a stylized abstraction.               |
| Strip has uneven frame widths        | Ask for `"evenly spaced, identical frame widths"`; re-generate if stuck.   |

## Output conventions

- Always PNG with alpha.
- Filenames: `<subject>_<pose>.png`, `<subject>_walk_<00>.png`, `<subject>_sheet.png`.
- For sprite sheets, write the individual frames next to the sheet so the
  engine author can pick either.
- All output paths must live inside your workspace (`workspaces/sprite_maker/`).
  Use relative paths like `robot.png` or `frames/walk_00.png` when
  invoking the helper.

## Delivering outputs to the user

Write files to your workspace, then surface them with `<send_file>`:

```
<send_file>robot_walk_sheet.png</send_file>
<send_file>frames/walk_00.png</send_file>
<send_file>frames/walk_01.png</send_file>
```

- One tag per file; paths are workspace-relative.
- Tags are stripped from your visible reply before the user sees it, so
  don't also write "here's the sprite: robot.png" in prose — just use
  the tag and let the attachment speak.
- Surface only final outputs, not intermediates. If you regenerated three
  times before getting a good sprite, attach just the keeper.
- Hard cap: 16 attachments per reply, 25 MiB per file. Downscale with
  `--size` if you're close to the limit.

## When to push back

- Nano Banana struggles with: legible text on sprites, photoreal faces,
  tiny details (< 8px features in a larger sprite), very specific
  licensed IP. If the user asks for these, say so and suggest a simpler
  aesthetic.
- If the user wants genuinely hand-authored pixel art (not AI
  approximation), this tool is the wrong tool. Say so.
