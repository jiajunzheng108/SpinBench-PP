import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

G = np.array([0.0, 0.0, -9.81])
R = 0.02
M = 2.7e-3
KD0 = 3.8e-4
KM0 = 4.86e-6
DT = 0.0056


def read_trajectory(path):
    with open(path, "r", encoding="utf-8-sig") as fh:
        header = [c.strip() for c in fh.readline().strip().split(",")]
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    header_ok = len(header) >= 4 and header[0].lower() == "timestamp"
    return header, header_ok, arr[:, 0].astype(float), arr[:, 1:4].astype(float)


def detect_first_bounce(t, p, thr=0.06, rise=0.03, win=6):
    z = p[:, 2]
    for i in range(1, len(z) - 1):
        if z[i] >= thr:
            continue
        if not (z[i] <= z[i - 1] and z[i] <= z[i + 1]):
            continue
        j = min(len(z) - 1, i + win)
        if np.max(z[i + 1:j + 1]) - z[i] >= rise:
            return i
    if len(z) and float(np.min(z)) < thr:
        return int(np.argmin(z))
    return None


def resample_uniform(t, p, dt):
    T = t[-1] - t[0]
    if T <= 0:
        return None, None
    n = int(np.floor(T / dt)) + 1
    if n < 5:
        return None, None
    tg = np.arange(n) * dt
    base = t - t[0]
    pg = np.column_stack([np.interp(tg, base, p[:, i]) for i in range(3)])
    return tg, pg


def smooth_and_derivs(p, dt):
    n = len(p)
    wp = min(9, n if n % 2 == 1 else n - 1)
    if wp % 2 == 0:
        wp -= 1
    if wp < 5:
        wp = 5 if n >= 5 else n
        if wp % 2 == 0:
            wp -= 1
    if wp >= 5:
        ps = np.column_stack([savgol_filter(p[:, i], wp, 2, mode="interp") for i in range(3)])
    else:
        ps = p.copy()
    v = np.gradient(ps, dt, axis=0)
    wa = min(11, n if n % 2 == 1 else n - 1)
    if wa % 2 == 0:
        wa -= 1
    if wa < max(wp, 5):
        wa = max(wp, 5)
    if wa % 2 == 0:
        wa += 1
    if wa <= n:
        a = np.column_stack([savgol_filter(ps[:, i], wa, 2, deriv=2, delta=dt, mode="interp") for i in range(3)])
    else:
        a = np.gradient(v, dt, axis=0)
    return ps, v, a


def linear_spin_guess(v, a, kd, km):
    speed = np.linalg.norm(v, axis=1)
    b = (a - G + (kd / M) * (speed[:, None] * v)) * (M / km)
    rows = []
    for i in range(len(v)):
        vx, vy, vz = v[i]
        rows.append(np.array([[0.0, vz, -vy], [-vz, 0.0, vx], [vy, -vx, 0.0]]))
    A = np.concatenate(rows, axis=0)
    w, *_ = np.linalg.lstsq(A, b.reshape(-1), rcond=None)
    return w


def accel(v, w, kd, km):
    speed = np.linalg.norm(v)
    return G - (kd / M) * speed * v + (km / M) * np.cross(w, v)


def rk4_positions(r0, v0, w, n, dt, kd, km):
    r = np.empty((n, 3))
    v = np.asarray(v0, dtype=float).copy()
    r[0] = r0
    for i in range(n - 1):
        k1v = accel(v, w, kd, km)
        k1r = v
        k2v = accel(v + 0.5 * dt * k1v, w, kd, km)
        k2r = v + 0.5 * dt * k1v
        k3v = accel(v + 0.5 * dt * k2v, w, kd, km)
        k3r = v + 0.5 * dt * k2v
        k4v = accel(v + dt * k3v, w, kd, km)
        k4r = v + dt * k3v
        v = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        r[i + 1] = r[i] + dt / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
    return r


def fit_spin(tg, pg, kd, km, warm_start=None, max_nfev=80):
    n = len(pg)
    dt = float(tg[1] - tg[0])
    ps, v, a = smooth_and_derivs(pg, dt)
    wg = linear_spin_guess(v, a, kd, km)
    r0 = pg[0]
    v0 = v[0]

    def residual(x):
        return (rk4_positions(r0, x[3:6], x[:3], n, dt, kd, km) - pg).ravel()

    lb = np.array([-1000.0, -1000.0, -1000.0, -25.0, -25.0, -25.0])
    ub = np.array([1000.0, 1000.0, 1000.0, 25.0, 25.0, 25.0])

    starts = []
    if warm_start is not None:
        starts.append(np.r_[warm_start[0], warm_start[1]])
    starts.append(np.r_[wg, v0])
    starts.append(np.r_[np.zeros(3), v0])

    best = None
    for s in starts:
        s = np.clip(s, lb + 1e-8, ub - 1e-8)
        try:
            res = least_squares(residual, s, bounds=(lb, ub), max_nfev=max_nfev)
        except Exception:
            continue
        if best is None or res.cost < best.cost:
            best = res

    if best is None:
        return None

    def evaluate(x):
        rp = rk4_positions(r0, x[3:6], x[:3], n, dt, kd, km)
        rmse = float(np.sqrt(np.mean(np.sum((rp - pg) ** 2, axis=1))))
        vp = np.gradient(rp, dt, axis=0)
        v_rmse = float(np.sqrt(np.mean(np.sum((vp - v) ** 2, axis=1))))
        return rmse, v_rmse, rp

    rmse, v_rmse, rp = evaluate(best.x)

    if rmse > 0.02:
        extras = [
            np.r_[300.0 * np.array([0.0, 0.0, 1.0]), v0],
            np.r_[-300.0 * np.array([0.0, 0.0, 1.0]), v0],
            np.r_[300.0 * np.array([0.0, 1.0, 0.0]), v0],
            np.r_[300.0 * np.array([1.0, 0.0, 0.0]), v0],
        ]
        for s in extras:
            s = np.clip(s, lb + 1e-8, ub - 1e-8)
            try:
                res = least_squares(residual, s, bounds=(lb, ub), max_nfev=max_nfev)
            except Exception:
                continue
            if res.cost < best.cost:
                best = res
        rmse, v_rmse, rp = evaluate(best.x)

    return {
        "omega": best.x[:3],
        "v0": best.x[3:6],
        "rmse": rmse,
        "v_rmse": v_rmse,
        "success": bool(best.success),
        "nfev": int(best.nfev),
        "r_pred": rp,
    }


def estimate_kd(records, km):
    num = 0.0
    den = 0.0
    for r in records:
        tg = r["t"]
        pg = r["p"]
        w = r["omega"]
        dt = float(tg[1] - tg[0])
        ps, v, a = smooth_and_derivs(pg, dt)
        speed = np.linalg.norm(v, axis=1)
        magnus = (km / M) * np.cross(w, v)
        y = (a - G - magnus).reshape(-1)
        x = (-(1.0 / M) * (speed[:, None] * v)).reshape(-1)
        num += float(np.dot(x, y))
        den += float(np.dot(x, x))
    if den <= 0:
        return None
    return num / den


def stratified_split(records, seed=42):
    import random

    usable = [r for r in records if not r["unusable"]]
    if not usable:
        for r in records:
            r["split"] = "drop"
        return

    spins = np.array([r["omega_norm"] for r in usable])
    speeds = np.array([np.linalg.norm(r["v0"]) for r in usable])
    spin_edges = np.quantile(spins, [1 / 3, 2 / 3])
    speed_edges = np.quantile(speeds, [1 / 2])
    spin_bin = np.digitize(spins, spin_edges)
    speed_bin = np.digitize(speeds, speed_edges)
    labels = [f"{s}_{v}" for s, v in zip(spin_bin, speed_bin)]

    groups = {}
    for idx, label in enumerate(labels):
        groups.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    split = {}
    for label, idxs in groups.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        n_test = n - n_train - n_val
        if n >= 7:
            n_val = max(1, n_val)
            n_test = max(1, n_test)
            n_train = n - n_val - n_test
        for k, idx in enumerate(idxs):
            if k < n_train:
                split[idxs[k]] = "train"
            elif k < n_train + n_val:
                split[idxs[k]] = "val"
            else:
                split[idxs[k]] = "test"

    for i, r in enumerate(usable):
        r["split"] = split.get(i, "train")
    for r in records:
        if "split" not in r:
            r["split"] = "drop"


def process_one(args):
    path_str, dt, kd, km = args
    path = Path(path_str)
    header, header_ok, t, p = read_trajectory(path)
    arr_ok = np.isfinite(t).all() and np.isfinite(p).all()
    flags = []
    if not header_ok:
        flags.append("bad_header")
    if not arr_ok:
        flags.append("nonfinite")
    if len(t) < 5:
        return None

    order = np.argsort(t)
    t = t[order]
    p = p[order]
    if np.any(np.diff(t) <= 0):
        unique = np.concatenate([[True], np.diff(t) > 0])
        t = t[unique]
        p = p[unique]
        flags.append("duplicate_or_nonmonotonic")

    dt_all = np.diff(t)
    dt_med = float(np.median(dt_all)) if len(dt_all) else float("nan")
    dt_min = float(np.min(dt_all)) if len(dt_all) else float("nan")
    dt_max = float(np.max(dt_all)) if len(dt_all) else float("nan")
    gap_count = int(np.sum(dt_all > 2.5 * dt_med)) if len(dt_all) and dt_med > 0 else 0
    if gap_count:
        flags.append("gap")

    speed = np.linalg.norm(np.diff(p, axis=0) / dt_all[:, None], axis=1) if len(dt_all) else np.array([])
    speed_med = float(np.median(speed)) if len(speed) else float("nan")
    speed_max = float(np.max(speed)) if len(speed) else float("nan")
    if len(speed) and speed_max > 30:
        flags.append("speed_spike")

    z = p[:, 2]
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    below = int(np.sum(z < 0))
    if below:
        flags.append("below_table")

    bi = detect_first_bounce(t, p)
    if bi is None:
        flags.append("no_bounce")
        t_pre = t
        p_pre = p
        bi_use = len(t)
    else:
        if bi < 2:
            flags.append("bounce_too_early")
        t_pre = t[:bi]
        p_pre = p[:bi]
        bi_use = bi

    if len(t_pre) < 20:
        flags.append("short_pre")
    pre_duration = float(t_pre[-1] - t_pre[0]) if len(t_pre) > 1 else 0.0
    if pre_duration < 0.15:
        flags.append("short_pre_duration")
    pre_path = float(np.sum(np.linalg.norm(np.diff(p_pre, axis=0), axis=1))) if len(p_pre) > 1 else 0.0

    tg, pg = resample_uniform(t_pre, p_pre, dt)
    if tg is None:
        return None

    fit = fit_spin(tg, pg, kd, km)
    if fit is None:
        flags.append("fit_failed")
        fit = {
            "omega": np.zeros(3),
            "v0": np.zeros(3),
            "rmse": float("nan"),
            "v_rmse": float("nan"),
            "success": False,
            "nfev": 0,
            "r_pred": pg,
        }

    omega = fit["omega"]
    omega_norm = float(np.linalg.norm(omega))
    rmse = fit["rmse"]
    v_rmse = fit["v_rmse"]
    if not np.isfinite(rmse) or rmse > 0.02:
        flags.append("fit_bad")
    if omega_norm > 950:
        flags.append("omega_near_bound")
    if np.max(np.abs(fit["v0"])) > 24:
        flags.append("v0_near_bound")

    unusable = any(
        f in flags
        for f in [
            "short_pre",
            "short_pre_duration",
            "fit_failed",
            "fit_bad",
            "nonfinite",
        ]
    )

    audit = {
        "id": int(path.stem),
        "file": path.name,
        "header_ok": bool(header_ok),
        "n_points": int(len(t)),
        "dt_median": dt_med,
        "dt_min": dt_min,
        "dt_max": dt_max,
        "gap_count": gap_count,
        "t_start": float(t[0]),
        "t_end": float(t[-1]),
        "duration": float(t[-1] - t[0]),
        "x_min": float(p[:, 0].min()),
        "x_max": float(p[:, 0].max()),
        "y_min": float(p[:, 1].min()),
        "y_max": float(p[:, 1].max()),
        "z_min": z_min,
        "z_max": z_max,
        "below_table_count": below,
        "speed_median": speed_med,
        "speed_max": speed_max,
        "bounce_index": None if bi is None else int(bi),
        "pre_points": int(len(t_pre)),
        "pre_duration": pre_duration,
        "pre_path": pre_path,
        "flags": flags,
        "unusable": bool(unusable),
    }

    return {
        "id": int(path.stem),
        "path": str(path),
        "t": tg,
        "p": pg,
        "p_pre": p_pre,
        "omega": omega,
        "v0": fit["v0"],
        "omega_norm": omega_norm,
        "rmse": rmse,
        "v_rmse": v_rmse,
        "fit_success": fit["success"],
        "nfev": fit["nfev"],
        "r_pred": fit["r_pred"],
        "audit": audit,
        "unusable": bool(unusable),
        "flags": flags,
    }


def write_clean_csv(path, t, p):
    arr = np.column_stack([t, p])
    np.savetxt(path, arr, delimiter=",", header="Timestamp,X,Y,Z", comments="", fmt="%.6f")


def write_meta_csv(path, records, kd, km):
    fields = [
        "id",
        "split",
        "unusable",
        "n_pre",
        "n_resampled",
        "duration",
        "pre_path_m",
        "r0_x",
        "r0_y",
        "r0_z",
        "v0_x",
        "v0_y",
        "v0_z",
        "omega_x",
        "omega_y",
        "omega_z",
        "omega_norm",
        "rmse_cm",
        "v_rmse_cm",
        "kd",
        "km",
        "fit_success",
        "nfev",
        "landing_x",
        "landing_y",
        "landing_z",
        "flags",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sorted(records, key=lambda x: x["id"]):
            a = r["audit"]
            p_pre = r["p_pre"]
            landing = p_pre[-1] if len(p_pre) else np.array([np.nan, np.nan, np.nan])
            w.writerow({
                "id": r["id"],
                "split": r.get("split", "drop"),
                "unusable": int(r["unusable"]),
                "n_pre": a["pre_points"],
                "n_resampled": len(r["t"]),
                "duration": a["pre_duration"],
                "pre_path_m": a["pre_path"],
                "r0_x": r["p"][0, 0],
                "r0_y": r["p"][0, 1],
                "r0_z": r["p"][0, 2],
                "v0_x": r["v0"][0],
                "v0_y": r["v0"][1],
                "v0_z": r["v0"][2],
                "omega_x": r["omega"][0],
                "omega_y": r["omega"][1],
                "omega_z": r["omega"][2],
                "omega_norm": r["omega_norm"],
                "rmse_cm": r["rmse"] * 100.0 if np.isfinite(r["rmse"]) else float("nan"),
                "v_rmse_cm": r["v_rmse"] * 100.0 if np.isfinite(r["v_rmse"]) else float("nan"),
                "kd": kd,
                "km": km,
                "fit_success": int(r["fit_success"]),
                "nfev": r["nfev"],
                "landing_x": landing[0],
                "landing_y": landing[1],
                "landing_z": landing[2],
                "flags": ";".join(r["flags"]),
            })


def plot_diagnostics(path, records, kd, km):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    good = [r for r in records if np.isfinite(r["rmse"])]
    if not good:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes[0, 0]
    ax.hist([r["omega_norm"] for r in good], bins=20, color="#3964fe", alpha=0.8)
    ax.set_title("Spin magnitude |omega| (rad/s)")
    ax.set_xlabel("|omega|")

    ax = axes[0, 1]
    ax.hist([r["rmse"] * 100 for r in good], bins=20, color="#2ca02c", alpha=0.8)
    ax.set_title("Position RMSE (cm)")
    ax.set_xlabel("cm")

    ax = axes[0, 2]
    ax.scatter([r["omega"][0] for r in good], [r["omega"][1] for r in good], c=[r["omega_norm"] for r in good], s=18)
    ax.set_title("omega_x vs omega_y")
    ax.set_xlabel("omega_x")
    ax.set_ylabel("omega_y")

    ax = axes[1, 0]
    ax.hist([np.linalg.norm(r["v0"]) for r in good], bins=20, color="#ff7f0e", alpha=0.8)
    ax.set_title("Initial speed |v0| (m/s)")
    ax.set_xlabel("m/s")

    ax = axes[1, 1]
    ax.scatter([r["audit"]["pre_duration"] for r in good], [r["omega_norm"] for r in good], s=18)
    ax.set_title("Pre-bounce duration vs |omega|")
    ax.set_xlabel("duration (s)")
    ax.set_ylabel("|omega|")

    ax = axes[1, 2]
    chosen = sorted(good, key=lambda r: abs(r["rmse"] - np.median([x["rmse"] for x in good])))[:3]
    for r in chosen:
        dt = float(r["t"][1] - r["t"][0])
        rp = rk4_positions(r["p"][0], r["v0"], r["omega"], len(r["t"]), dt, kd, km)
        ax.plot(r["t"], r["p"][:, 2], label=f"{r['id']} obs", alpha=0.8)
        ax.plot(r["t"], rp[:, 2], "--", label=f"{r['id']} fit", alpha=0.8)
    ax.set_title("Example Z trajectories")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("z (m)")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="3D_gt")
    ap.add_argument("--output", default="prep_3d_gt/output")
    ap.add_argument("--dt", type=float, default=DT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--calib-iters", type=int, default=1)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    input_dir = Path(args.input)
    if not input_dir.is_absolute():
        input_dir = root / input_dir
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    clean_dir = output_dir / "3D_gt_clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(input_dir.glob("*.csv"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no csv files in {input_dir}")

    print(f"processing {len(paths)} files")
    records = []
    for i, path in enumerate(paths, 1):
        r = process_one((str(path), args.dt, KD0, KM0))
        if r is None:
            print(f"  skip {path.name}")
            continue
        records.append(r)
        if i % 20 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)}  last id={r['id']} rmse={r['rmse']*100:.2f}cm |w|={r['omega_norm']:.1f}")

    stratified_split(records, seed=args.seed)

    kd = KD0
    km = KM0
    calib = []
    for it in range(max(0, args.calib_iters)):
        train_records = [r for r in records if r.get("split") == "train" and not r["unusable"]]
        kd_est = estimate_kd(train_records, km)
        if kd_est is None:
            break
        kd_est = float(np.clip(kd_est, 0.5 * KD0, 2.0 * KD0))
        before = {}
        after = {}
        for split_name in ("train", "val"):
            subset = [r for r in records if r.get("split") == split_name and not r["unusable"]]
            if subset:
                before[split_name] = float(np.mean([r["rmse"] for r in subset]))
        kd = kd_est
        for r in records:
            fit = fit_spin(r["t"], r["p"], kd, km, warm_start=(r["omega"], r["v0"]))
            if fit is not None:
                r["omega"] = fit["omega"]
                r["v0"] = fit["v0"]
                r["omega_norm"] = float(np.linalg.norm(fit["omega"]))
                r["rmse"] = fit["rmse"]
                r["v_rmse"] = fit["v_rmse"]
                r["fit_success"] = fit["success"]
                r["nfev"] = fit["nfev"]
                r["r_pred"] = fit["r_pred"]
        for split_name in ("train", "val"):
            subset = [r for r in records if r.get("split") == split_name and not r["unusable"]]
            if subset:
                after[split_name] = float(np.mean([r["rmse"] for r in subset]))
        calib.append({
            "iteration": it + 1,
            "kd_before": KD0 if it == 0 else calib[-1]["kd_after"],
            "kd_after": kd,
            "km": km,
            "rmse_before_cm": {k: v * 100 for k, v in before.items()},
            "rmse_after_cm": {k: v * 100 for k, v in after.items()},
        })
        print(f"  calib iter {it+1}: KD {calib[-1]['kd_before']:.6g} -> {kd:.6g}")

    for r in records:
        w = write_clean_csv(clean_dir / f"{r['id']:03d}.csv", r["t"], r["p"])

    meta_path = output_dir / "3D_gt_meta.csv"
    write_meta_csv(meta_path, records, kd, km)

    splits_path = output_dir / "3D_gt_splits.csv"
    with open(splits_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "split", "unusable"])
        for r in sorted(records, key=lambda x: x["id"]):
            w.writerow([r["id"], r.get("split", "drop"), int(r["unusable"])])

    audit = {
        "summary": {
            "n_files": len(records),
            "n_usable": int(sum(1 for r in records if not r["unusable"])),
            "n_unusable": int(sum(1 for r in records if r["unusable"])),
            "split_counts": {k: int(sum(1 for r in records if r.get("split") == k)) for k in ("train", "val", "test", "drop")},
            "dt_median_all": float(np.median([r["audit"]["dt_median"] for r in records])),
            "points_median": float(np.median([r["audit"]["n_points"] for r in records])),
            "pre_points_median": float(np.median([r["audit"]["pre_points"] for r in records])),
            "pre_duration_median": float(np.median([r["audit"]["pre_duration"] for r in records])),
            "pre_path_median": float(np.median([r["audit"]["pre_path"] for r in records])),
            "omega_norm_median": float(np.median([r["omega_norm"] for r in records])),
            "omega_norm_p95": float(np.percentile([r["omega_norm"] for r in records], 95)),
            "omega_norm_max": float(np.max([r["omega_norm"] for r in records])),
            "rmse_cm_median": float(np.median([r["rmse"] * 100 for r in records])),
            "rmse_cm_p95": float(np.percentile([r["rmse"] * 100 for r in records], 95)),
            "flag_counts": {},
        },
        "per_file": [r["audit"] for r in sorted(records, key=lambda x: x["id"])],
    }
    for r in records:
        for f in r["flags"]:
            audit["summary"]["flag_counts"][f] = audit["summary"]["flag_counts"].get(f, 0) + 1

    with open(output_dir / "audit_report.json", "w", encoding="utf-8") as fh:
        json.dump(audit, fh, ensure_ascii=False, indent=2)

    calib_out = {
        "kd_initial": KD0,
        "km_fixed": km,
        "km_note": "KM is not identifiable separately from spin magnitude; spin values are defined at this fixed KM.",
        "iterations": calib,
    }
    with open(output_dir / "calibration.json", "w", encoding="utf-8") as fh:
        json.dump(calib_out, fh, ensure_ascii=False, indent=2)

    if not args.no_plot:
        plot_diagnostics(output_dir / "diagnostics.png", records, kd, km)

    s = audit["summary"]
    print("\nsummary")
    print(f"  files: {s['n_files']}  usable: {s['n_usable']}  unusable: {s['n_unusable']}")
    print(f"  split: {s['split_counts']}")
    print(f"  pre-bounce points median: {s['pre_points_median']:.0f}  duration median: {s['pre_duration_median']:.3f}s")
    print(f"  |omega| median/p95/max: {s['omega_norm_median']:.1f} / {s['omega_norm_p95']:.1f} / {s['omega_norm_max']:.1f} rad/s")
    print(f"  rmse median/p95: {s['rmse_cm_median']:.2f} / {s['rmse_cm_p95']:.2f} cm")
    print(f"  flags: {s['flag_counts']}")
    print(f"  KD: {KD0:.6g} -> {kd:.6g}   KM fixed: {km:.6g}")
    print(f"  output: {output_dir}")


if __name__ == "__main__":
    main()
