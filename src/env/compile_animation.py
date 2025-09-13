#!/usr/bin/env python3
"""
Standalone animation compiler (no CLI args).

Edit the CONFIG block in main() and run:
  python compile_animation.py

It looks for frames like t0000.png, t0001.png, ... in FRAMES_DIR and writes
animation.gif and animation.mp4 there (best-effort).
"""

from __future__ import annotations

import os
import sys
import glob
import re
from typing import List, Optional


def _collect_frames(frames_dir: str, pattern: str, recursive: bool = False) -> List[str]:
    search = os.path.join(frames_dir, pattern)
    paths = glob.glob(search, recursive=recursive)
    if not paths:
        return []

    def _sort_key(p: str):
        b = os.path.basename(p)
        m = re.search(r"(\d+)", b)
        return (int(m.group(1)) if m else 0, b.lower())

    paths.sort(key=_sort_key)
    return paths


def _compile_gif(paths: List[str], out_path: str, fps: int, loop: int = 0) -> bool:
    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        print(f"[warn] GIF export skipped (PIL not available): {e}")
        return False

    if not paths:
        print("[warn] No frames to write GIF.")
        return False

    try:
        imgs = [Image.open(p) for p in paths]
        duration = max(1, int(1000 / max(1, fps)))  # ms per frame
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        imgs[0].save(
            out_path,
            save_all=True,
            append_images=imgs[1:],
            duration=duration,
            loop=loop,
            optimize=False,
        )
        for im in imgs:
            try:
                im.close()
            except Exception:
                pass
        print(f"[ok] Wrote GIF: {out_path}")
        return True
    except Exception as e:
        print(f"[warn] GIF export failed: {e}")
        return False


def _compile_mp4(paths: List[str], out_path: str, fps: int) -> bool:
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception as e:
        print(f"[warn] MP4 export skipped (imageio not available): {e}")
        return False

    if not paths:
        print("[warn] No frames to write MP4.")
        return False

    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with imageio.get_writer(out_path, fps=max(1, int(fps))) as writer:
            for p in paths:
                writer.append_data(imageio.imread(p))
        print(f"[ok] Wrote MP4: {out_path}")
        return True
    except Exception as e:
        print(f"[warn] MP4 export failed: {e}")
        return False


def run(
    frames_dir: str,
    pattern: str = "t*.png",
    fps: int = 10,
    step: int = 1,
    reverse: bool = False,
    do_gif: bool = True,
    do_mp4: bool = True,
    gif_path: Optional[str] = None,
    mp4_path: Optional[str] = None,
) -> int:
    frames_dir = os.path.abspath(frames_dir)
    if not os.path.isdir(frames_dir):
        print(f"[err] Not a directory: {frames_dir}")
        return 2

    paths = _collect_frames(frames_dir, pattern)
    if step > 1:
        paths = paths[:: max(1, int(step))]
    if reverse and paths:
        paths = paths + list(reversed(paths))

    print(f"[info] frames_dir={frames_dir}, frames={len(paths)}, fps={fps}, step={step}, reverse={reverse}")

    wrote_any = False
    if do_gif:
        gif_path = gif_path or os.path.join(frames_dir, "animation.gif")
        wrote_any = _compile_gif(paths, gif_path, fps=fps) or wrote_any
    if do_mp4:
        mp4_path = mp4_path or os.path.join(frames_dir, "animation.mp4")
        wrote_any = _compile_mp4(paths, mp4_path, fps=fps) or wrote_any

    if not wrote_any:
        print("[warn] Nothing was written. Check that frames exist and required libraries are installed (Pillow/imageio).")
        return 1
    return 0


def main() -> int:
    # ===== CONFIG (edit here) =====
    FRAMES_DIR = "results/S-Shape_ep00"  # e.g., results/S-Shape_ep00
    PATTERN = "t*.png"                   # glob pattern for frames
    FPS = 10                             # frames per second
    STEP = 10                             # take every Nth frame
    REVERSE = False                      # append reversed frames
    DO_GIF = True                        # write GIF
    DO_MP4 = True                        # write MP4
    GIF_PATH = None                      # None -> <FRAMES_DIR>/animation.gif
    MP4_PATH = None                      # None -> <FRAMES_DIR>/animation.mp4
    # ===== end CONFIG =====

    return run(
        frames_dir=FRAMES_DIR,
        pattern=PATTERN,
        fps=FPS,
        step=STEP,
        reverse=REVERSE,
        do_gif=DO_GIF,
        do_mp4=DO_MP4,
        gif_path=GIF_PATH,
        mp4_path=MP4_PATH,
    )


if __name__ == "__main__":
    sys.exit(main())
