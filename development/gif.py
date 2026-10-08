"""Animated view of the development tracker: for a few batters, the line the
aging model drew season by season through 2024, then where 2025 landed.

The filter is the one in tracker.py (same plug-in parameters); this script
re-runs it to keep the path, not just the end point, and checks that the end
point equals tracker_2025.csv for every batter it draws.

Usage: python gif.py <prepared dir> <final run dir> <mart_batter_season parquet> <tracker csv> <out gif>
"""
import csv
import io
import json
import pathlib
import sys

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

Z90 = 1.6448536269514722
PLAYERS = [672695, 668930, 671739, 672724]   # Perdomo, Turang, Harris II, Peraza
YLIM = (-0.12, 0.09)
MIN_DOT = 10   # a 11-PA season would otherwise be a speck
AHEAD, BEHIND, LINE, DOT = "#1f7a4d", "#b8432f", "#3b5b92", "#222222"


def filter_paths(d, p, cum):
    """Kalman filter of tracker.py, keeping (mean, var) before and after every row."""
    mf, vf, mp, vp = (np.empty(d["N"]) for _ in range(4))
    m = v = 0.0
    for n in range(d["N"]):
        if d["is_first"][n]:
            m = p["mu_entry"] + p["b_entry"] * d["entry_age_c"][n]
            v = p["sd_entry"] ** 2
        else:
            m += cum[d["age_idx"][n] - 1] - cum[d["prev_age_idx"][n] - 1]
            v += p["sd_step"] ** 2 * d["years"][n]
        mp[n], vp[n] = m, v
        r = p["sigma_pa"] ** 2 / d["pa"][n] + p["tau"] ** 2
        g = v / (v + r)
        m += g * (d["y"][n] - m)
        v *= 1 - g
        mf[n], vf[n] = m, v
    # the path drawn between seasons is the prior: check it is the predict step
    # of the previous posterior (the end-point check against the tracker csv
    # does not see the path)
    for n in range(1, d["N"]):
        if d["is_first"][n]:
            continue
        m_pred = mf[n - 1] + cum[d["age_idx"][n] - 1] - cum[d["prev_age_idx"][n] - 1]
        v_pred = vf[n - 1] + p["sd_step"] ** 2 * d["years"][n]
        if abs(mp[n] - m_pred) > 1e-12 or abs(vp[n] - v_pred) > 1e-12:
            raise SystemExit(f"row {n}: prior is not the predict step of the previous posterior")
    return mf, vf, mp, vp


def main(prep_dir, run_dir, mart, tracker_csv, out_gif):
    prep, run = pathlib.Path(prep_dir), pathlib.Path(run_dir)
    d = json.loads((prep / "stan_data.json").read_text())
    ev = json.loads((prep / "eval.json").read_text())
    f = json.loads((run / "fit_aging.json").read_text())
    p = {k: f[k]["mean"] for k in ("mu_entry", "b_entry", "sd_entry", "sd_step", "sigma_pa", "tau")}
    cum = np.array(f["curve"]["0.5"])
    mf, vf, mp, vp = filter_paths(d, p, cum)

    # the training rows in prepare.py's order, to recover each row's season
    con = duckdb.connect()
    con.execute("create table b as select * from read_parquet(?) where primary_position <> 'P'", [mart])
    con.execute("create table lg as select season, sum(woba * pa) / sum(pa) as lg_woba from b group by season")
    rows = con.execute("""select b.player_id, b.season, b.pa, b.woba - lg.lg_woba from b join lg using (season)
                          where b.season <= 2024 order by b.player_id, b.season""").fetchall()
    if len(rows) != d["N"] or any(abs(r[3] - y) > 1e-12 or r[2] != pa for r, y, pa in zip(rows, d["y"], d["pa"])):
        raise SystemExit("training rows do not line up with stan_data.json")
    track = {int(r["player_id"]): r for r in csv.DictReader(open(tracker_csv, encoding="utf-8"))}

    panels = []
    for pid in PLAYERS:
        j = next(i for i, e in enumerate(ev) if e["player_id"] == pid)
        e = ev[j]
        idx = [n for n, r in enumerate(rows) if r[0] == pid]
        last = e["last_obs"] - 1
        if idx[-1] != last:
            raise SystemExit(f"{pid}: last training row mismatch")
        gap = e["target_age_idx"] - d["age_idx"][last]
        exp_ = mf[last] + cum[e["target_age_idx"] - 1] - cum[d["age_idx"][last] - 1]
        sd = float(np.sqrt(vf[last] + p["sd_step"] ** 2 * gap + p["tau"] ** 2
                           + p["sigma_pa"] ** 2 / e["pa"] + f["pred_sd"][j] ** 2))
        z = (e["target"] - exp_) / sd
        t = track[pid]
        if abs(round(exp_, 4) - float(t["expected_rel_woba"])) > 0 or abs(round(z, 3) - float(t["z"])) > 0 \
                or abs(round(sd, 4) - float(t["sd"])) > 0:
            raise SystemExit(f"{pid}: end point differs from tracker csv")
        panels.append({
            "name": e["player_name"], "age": e["age"], "z": z,
            "seasons": [rows[n][1] for n in idx], "y": [rows[n][3] for n in idx], "pa": [rows[n][2] for n in idx],
            "m": [mf[n] for n in idx], "s": [np.sqrt(vf[n]) for n in idx],
            "mp": [mp[n] for n in idx], "sp": [np.sqrt(vp[n]) for n in idx],
            "exp": exp_, "sd": sd, "actual": e["target"], "pa25": e["pa"]})

    first = min(s for q in panels for s in q["seasons"])
    # fixed limits: limits taken from the data would let the 2025 actual move
    # the axis before its dot appears
    ymin, ymax = YLIM
    for q in panels:
        lo = min(min(q["y"]), q["exp"] - Z90 * q["sd"], q["actual"], min(np.subtract(q["m"], Z90 * np.array(q["s"]))))
        hi = max(max(q["y"]), q["exp"] + Z90 * q["sd"], q["actual"], max(np.add(q["m"], Z90 * np.array(q["s"]))))
        if lo < ymin or hi > ymax:
            raise SystemExit(f"{q['name']}: data outside the fixed y limits")

    # frames: the line grows continuously through 2024 (held at each season),
    # then the dashed projection reaches 2025, then the band, then 2025 lands
    steps, durations = [], []
    for yr in range(first, 2025):
        for t in (np.linspace(0.2, 1.0, 5) if yr > first else [1.0]):
            steps.append((yr - 1 + t, 0, 1.0))
            durations.append(700 if t == 1.0 else 70)
    for t in np.linspace(0.2, 1.0, 5):
        steps.append((2024.0, 1, t))
        durations.append(70)
    durations[-1] = 1500
    steps.append((2024.0, 2, 1.0))
    durations.append(5000)
    frames = []
    for x_end, stage, frac in steps:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), dpi=80, sharex=True, sharey=True)
        fig.patch.set_facecolor("white")
        for ax, q in zip(axes.flat, panels):
            ax.set_facecolor("white")
            ax.axhline(0, color="#999999", lw=1)
            S = np.array(q["seasons"], float)
            M, SD = np.array(q["m"]), np.array(q["s"])
            if S[0] <= x_end + 1e-9:
                # seasons already seen: the estimate after each; between seasons the
                # line heads for the prediction made before the next season is seen
                xe = min(x_end, S[-1])
                k = int(np.sum(S <= xe + 1e-9))
                xs, ms, ss = list(S[:k]), list(M[:k]), list(SD[:k])
                if k < len(S) and xe > S[k - 1] + 1e-9:
                    t = (xe - S[k - 1]) / (S[k] - S[k - 1])
                    xs.append(xe)
                    ms.append(M[k - 1] + t * (q["mp"][k] - M[k - 1]))
                    ss.append(SD[k - 1] + t * (q["sp"][k] - SD[k - 1]))
                xs, ms, ss = np.array(xs), np.array(ms), np.array(ss)
                ax.fill_between(xs, ms - Z90 * ss, ms + Z90 * ss, color=LINE, alpha=0.15, lw=0)
                ax.plot(xs, ms, color=LINE, lw=2.5)
                ax.scatter(S[S <= xe + 1e-9], M[S <= xe + 1e-9], s=18, color=LINE, zorder=3)
                done = S <= x_end + 1e-9
                ax.scatter(S[done], np.array(q["y"])[done], s=np.maximum(np.array(q["pa"])[done] / 5, MIN_DOT), color=DOT, zorder=4)
            if stage >= 1:
                x1 = S[-1] + (2025 - S[-1]) * frac
                y1 = q["m"][-1] + (q["exp"] - q["m"][-1]) * frac
                ax.plot([S[-1], x1], [q["m"][-1], y1], color=LINE, lw=2.5, ls="--")
                if frac == 1.0:
                    ax.errorbar([2025], [q["exp"]], yerr=[[Z90 * q["sd"]]], color=LINE, lw=3, capsize=9)
            title = f"{q['name']} (age {q['age']} in 2025)"
            if stage == 2:
                col = AHEAD if q["z"] > 0 else BEHIND
                ax.scatter([2025], [q["actual"]], s=max(q["pa25"] / 5, MIN_DOT), color=col, zorder=5, edgecolor="white", lw=1.5)
                ax.set_title(title + f"   z = {q['z']:+.2f}", fontsize=13, color=col, loc="left")
            else:
                ax.set_title(title, fontsize=13, loc="left")
            ax.set_xlim(first - 0.5, 2025.6)
            ax.set_ylim(ymin, ymax)
            ax.set_xticks(range(first, 2026))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 1000:+.0f}"))
            ax.tick_params(labelsize=11)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        fig.supylabel("wOBA vs league (points, 1 pt = .001)", fontsize=13)
        head = ("2025 landed outside the 90% forecast: two ahead, two behind" if stage == 2
                else "The line the aging model drew for each batter, season by season")
        fig.suptitle(head, fontsize=16, x=0.02, ha="left", y=0.985)
        label = {0: str(int(np.floor(x_end + 1e-9))), 1: "2025 forecast", 2: "2025 actual"}[stage]
        fig.text(0.98, 0.94, label, fontsize=24, ha="right", va="top", weight="bold", color="#444444")
        fig.text(0.02, 0.01, "Black dots: seasons (size = PA). Blue line and shade: the model's estimate of talent "
                 "and its 90% band. Bar at 2025: 90% forecast.", fontsize=10, color="#555555")
        fig.tight_layout(rect=(0, 0.03, 1, 0.885))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="white")
        plt.close(fig)
        frames.append(Image.open(buf).convert("RGB"))

    pal = frames[-1].quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    q[0].save(out_gif, save_all=True, append_images=q[1:], duration=durations, loop=0, optimize=True)
    print(json.dumps({"frames": len(q), "seconds": sum(durations) / 1000, "bytes": pathlib.Path(out_gif).stat().st_size,
                      "z": {pp["name"]: round(pp["z"], 3) for pp in panels}}))


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
