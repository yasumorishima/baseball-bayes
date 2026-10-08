"""Animated aging curve of the final fit: relative wOBA against the peak,
posterior median and 90% band, drawn one age at a time from 20 to 40.

Reads only fit_aging.json of the final run. The numbers it shows are checked
against the table in RESULTS.md before anything is drawn.

Usage: python curve_gif.py <final run dir> <out gif>
"""
import io
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

A0, A1 = 20, 40
LINE, BAND, MARK = "#3b5b92", "#3b5b92", "#b8432f"
# RESULTS.md, "Descriptive": age -> (median, 5%, 95%), rounded to 0.001
TABLE = {21: (-0.017, -0.023, -0.012), 24: (-0.003, -0.005, -0.001), 27: (-0.000, -0.001, 0.000),
         30: (-0.006, -0.008, -0.004), 33: (-0.019, -0.023, -0.016), 36: (-0.040, -0.046, -0.034),
         40: (-0.073, -0.088, -0.058)}


def main(run_dir, out_gif):
    f = json.loads((pathlib.Path(run_dir) / "fit_aging.json").read_text())
    ages = np.array(f["ages"])
    med, lo, hi = (np.array(f["curve"][k]) for k in ("0.5", "0.05", "0.95"))
    for a, vals in TABLE.items():
        i = int(np.where(ages == a)[0][0])
        got = tuple(round(float(x) + 0.0, 3) for x in (med[i], lo[i], hi[i]))
        if any(abs(g - v) > 1e-9 for g, v in zip(got, vals)):
            raise SystemExit(f"age {a}: fit gives {got}, RESULTS.md says {vals}")
    keep = (ages >= A0) & (ages <= A1)
    ages, med, lo, hi = ages[keep].astype(float), med[keep], lo[keep], hi[keep]
    peak = int(ages[np.argmax(med)])
    # RESULTS.md: the peak is 26 (posterior probability 0.52) or 27 (0.42)
    if peak not in (26, 27):
        raise SystemExit(f"median curve peaks at {peak}")

    # the line advances a quarter year per frame and holds on the peak and the end
    xs = np.round(np.arange(A0, A1 + 1e-9, 0.25), 2)
    durations = [60] * len(xs)
    durations[0] = 800
    durations[int(np.where(xs == peak)[0][0])] = 1400
    durations[-1] = 5000
    frames = []
    for x in xs:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.axhline(0, color="#999999", lw=1)
        sel = ages <= x + 1e-9
        gx = np.append(ages[sel], x)
        gm, gl, gh = (np.interp(gx, ages, v) for v in (med, lo, hi))
        ax.fill_between(gx, gl * 1000, gh * 1000, color=BAND, alpha=0.18, lw=0)
        ax.plot(gx, gm * 1000, color=LINE, lw=3)
        y = float(np.interp(x, ages, med)) * 1000
        ax.scatter([x], [y], s=90, color=MARK, zorder=4)
        yr = round(y)
        ax.annotate(f"{yr:+d} pts" if yr else "0 pts", (x, y), xytext=(12, -6), textcoords="offset points",
                    fontsize=16, color=MARK, weight="bold", ha="left")
        if x >= peak:
            ax.axvspan(26, 27, color="#999999", alpha=0.12, lw=0)
            ax.text(peak + 0.5, 6, "peak 26-27", ha="center", fontsize=12, color="#555555")
        ax.set_xlim(A0 - 0.5, A1 + 2.6)  # room for the label at 40
        ax.set_ylim(min(lo) * 1000 - 5, 12)
        ax.set_xticks(range(A0, A1 + 1, 2))
        ax.tick_params(labelsize=12)
        ax.set_xlabel("Age", fontsize=14)
        ax.set_ylabel("wOBA vs peak (points, 1 pt = .001)", fontsize=14)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        head = (f"MLB batters peak at 26-27 and lose about {abs(TABLE[36][0]) * 1000:.0f} points of wOBA by 36"
                if x == xs[-1] else "The aging curve of the final fit, one age at a time")
        fig.suptitle(head, fontsize=16, x=0.02, ha="left", y=0.975)
        fig.text(0.98, 0.9, f"age {int(np.floor(x + 1e-9))}", fontsize=26, ha="right", va="top",
                 weight="bold", color="#444444")
        fig.text(0.02, 0.01, "Line: posterior median. Shade: 90% band. Batters 2015-2024, relative to the "
                 "league wOBA of each season.", fontsize=10, color="#555555")
        fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.15)  # fixed, so frames do not jitter
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="white")
        plt.close(fig)
        frames.append(Image.open(buf).convert("RGB"))

    pal = frames[-1].quantize(colors=64, method=Image.Quantize.MEDIANCUT)  # last frame holds every colour
    q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    q[0].save(out_gif, save_all=True, append_images=q[1:], duration=durations, loop=0, optimize=True)
    print(json.dumps({"frames": len(q), "seconds": sum(durations) / 1000,
                      "bytes": pathlib.Path(out_gif).stat().st_size, "peak": peak}))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
