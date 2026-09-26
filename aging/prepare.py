"""Build the Stan data and the evaluation table for the 2025 projection test.

Reads mart_batter_season (published by yasumorishima/mlb-data-pipeline to the
Hugging Face dataset yasumorishima/mlb-stats under marts/). Pitchers
(primary_position = 'P') are left out. wOBA is expressed relative to the
PA-weighted league wOBA of the same season.

Training uses seasons <= TRAIN_END only. The evaluation set is every batter
with PA >= EVAL_MIN_PA in EVAL_SEASON who has at least one training season.
Marcel is computed here from training seasons only.

primary_position is the position of that season (it changes between seasons
for many players), so filtering on it uses no future information. 'X' rows
(bench players without a regular position) are kept.

Usage: python prepare.py <mart_batter_season parquet path or URL> <out dir>
                         [train_end eval_season]
The defaults are the pre-registered 2024 / 2025. The rehearsal uses 2023 / 2024.
"""
import json
import pathlib
import sys

import duckdb

EVAL_MIN_PA = 100
# Marcel (Tango): weights 5/4/3 on the last three seasons, 1200 PA of league
# average added, age factor 1 + (29 - age) * 0.006 below 29 and * 0.003 above.
MARCEL_WEIGHTS = {1: 5.0, 2: 4.0, 3: 3.0}
MARCEL_REGRESS_PA = 1200.0


def marcel_age_factor(age: int) -> float:
    return 1 + (29 - age) * (0.006 if age < 29 else 0.003)


def main(src: str, out_dir: str, TRAIN_END: int = 2024, EVAL_SEASON: int = 2025) -> None:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("create table b as select * from read_parquet(?) where primary_position <> 'P'", [src])
    con.execute("""
        create table lg as
        select season, sum(woba * pa) / sum(pa) as lg_woba from b group by season
    """)
    rows = con.execute("""
        select b.player_id, b.season, b.age, b.pa, b.woba - lg.lg_woba as y
        from b join lg using (season)
        where b.season <= ?
        order by b.player_id, b.season
    """, [TRAIN_END]).fetchall()
    ev = con.execute("""
        select e.player_id, e.player_name, e.age, e.pa, e.woba - lg.lg_woba as target
        from b e join lg using (season)
        where e.season = ? and e.pa >= ?
          and exists (select 1 from b t where t.player_id = e.player_id and t.season <= ?)
        order by e.player_id
    """, [EVAL_SEASON, EVAL_MIN_PA, TRAIN_END]).fetchall()
    lg = dict(con.execute("select season, lg_woba from lg").fetchall())

    a_min = min(min(r[2] for r in rows), min(e[2] for e in ev))
    a_max = max(max(r[2] for r in rows), max(e[2] for e in ev))
    K = a_max - a_min + 1

    age_idx, is_first, prev_age_idx, years, entry_age_c, y, pa = [], [], [], [], [], [], []
    last_row = {}
    for n, (pid, season, age, p, yy) in enumerate(rows):
        first = n == 0 or rows[n - 1][0] != pid
        if not first and age <= rows[n - 1][2]:
            raise SystemExit(f"age does not increase for player {pid} at {season}")
        age_idx.append(age - a_min + 1)
        is_first.append(int(first))
        prev_age_idx.append(0 if first else rows[n - 1][2] - a_min + 1)
        years.append(0.0 if first else float(season - rows[n - 1][1]))
        entry_age_c.append(float(age - 27) if first else 0.0)
        y.append(yy)
        pa.append(float(p))
        last_row[pid] = n

    # Marcel from training seasons only, on the relative scale (league = 0).
    by_player = {}
    for pid, season, age, p, yy in rows:
        by_player.setdefault(pid, {})[season] = (p, yy)
    lg_ref = sum(lg[TRAIN_END - k + 1] for k in MARCEL_WEIGHTS) / len(MARCEL_WEIGHTS)
    eval_rows = []
    for pid, name, age, p, target in ev:
        num = den = 0.0
        for back, w in MARCEL_WEIGHTS.items():
            s = by_player[pid].get(EVAL_SEASON - back)
            if s:
                num += w * s[0] * s[1]
                den += w * s[0]
        rel = num / (den + MARCEL_REGRESS_PA)
        marcel = (lg_ref + rel) * marcel_age_factor(age) - lg_ref
        eval_rows.append({
            "player_id": pid, "player_name": name, "age": age, "pa": p,
            "target": target, "marcel": marcel,
            "last_obs": last_row[pid] + 1, "target_age_idx": age - a_min + 1,
        })

    stan = {
        "N": len(rows), "K": K, "age_idx": age_idx, "is_first": is_first,
        "prev_age_idx": prev_age_idx, "years": years, "entry_age_c": entry_age_c, "y": y, "pa": pa,
        "anchor_idx": 27 - a_min + 1,
        "M": len(eval_rows),
        "last_obs": [e["last_obs"] for e in eval_rows],
        "target_age_idx": [e["target_age_idx"] for e in eval_rows],
    }
    (out / "stan_data.json").write_text(json.dumps(stan))
    (out / "eval.json").write_text(json.dumps(eval_rows, ensure_ascii=False, indent=1))
    meta = {"source": src, "train_end": TRAIN_END, "eval_season": EVAL_SEASON, "a_min": a_min, "a_max": a_max, "N": len(rows),
            "players": len(by_player), "M": len(eval_rows),
            "lg_woba": {str(k): v for k, v in sorted(lg.items())}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("a_min", "a_max", "N", "players", "M")}))


if __name__ == "__main__":
    if len(sys.argv) not in (3, 5):
        raise SystemExit(__doc__)
    main(*sys.argv[1:3], *map(int, sys.argv[3:5]))
