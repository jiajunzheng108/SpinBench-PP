import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

G = np.array([0.0, 0.0, -9.81])
R = 0.02
M = 2.7e-3
KD = 3.9827679117703966e-4
KM = 4.86e-6


def read_traj(path):
    with open(path, "r", encoding="utf-8-sig") as fh:
        first = fh.readline().strip()
    delimiter = ";" if ";" in first else ","
    skip = 1 if any(c.isalpha() for c in first) else 0
    arr = np.loadtxt(str(path), delimiter=delimiter, skiprows=skip)
    return arr[:, 0].astype(np.float64), arr[:, 1:4].astype(np.float64)


def first_bounce(t, p, thr=0.06, rise=0.03, win=6):
    z = p[:, 2]
    for i in range(1, len(z) - 1):
        if z[i] >= thr:
            continue
        if not (z[i] <= z[i - 1] and z[i] <= z[i + 1]):
            continue
        j = min(len(z) - 1, i + win)
        if np.max(z[i + 1:j + 1]) - z[i] >= rise:
            return i
    return None


def load_index(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        r = csv.reader(fh)
        next(r)
        for row in r:
            if len(row) < 4:
                continue
            out[int(row[0])] = np.array([float(row[1]), float(row[2]), float(row[3])])
    return out


def select_usable(traj_dir, min_points, min_duration):
    files = sorted([f for f in traj_dir.glob("*.csv") if f.name != "index.csv"])
    out = []
    for f in files:
        t, p = read_traj(f)
        bi = first_bounce(t, p)
        if bi is not None:
            if bi < 3:
                continue
            t_use, p_use = t[:bi], p[:bi]
        else:
            t_use, p_use = t, p
        n = len(t_use)
        dur = float(t_use[-1] - t_use[0]) if n > 1 else 0.0
        if n < min_points or dur < min_duration:
            continue
        dt = np.diff(t_use)
        if len(dt) and np.any(dt > 2.5 * np.median(dt)):
            continue
        out.append({"id": int(f.stem), "t": t_use, "p": p_use})
    return out


def smooth_derivs(t, p):
    n = len(t)
    dt = float(np.median(np.diff(t)))
    w = min(11, n if n % 2 == 1 else n - 1)
    if w % 2 == 0:
        w -= 1
    if w < 5:
        w = 5 if n >= 5 else n
        if w % 2 == 0:
            w -= 1
    if w >= 5:
        ps = np.column_stack([savgol_filter(p[:, j], w, 2, mode="interp") for j in range(3)])
        v = np.gradient(ps, t, axis=0)
        a = np.column_stack([savgol_filter(ps[:, j], w, 2, deriv=2, delta=dt, mode="interp") for j in range(3)])
    else:
        ps = p.copy()
        v = np.gradient(ps, t, axis=0)
        a = np.gradient(v, t, axis=0)
    return ps, v, a


def linear_omega(v, a):
    speed = np.linalg.norm(v, axis=1)
    b = (a - G + (KD / M) * (speed[:, None] * v)) * (M / KM)
    rows = []
    for i in range(len(v)):
        vx, vy, vz = v[i]
        rows.append(np.array([[0.0, vz, -vy], [-vz, 0.0, vx], [vy, -vx, 0.0]]))
    A = np.concatenate(rows, axis=0)
    w, *_ = np.linalg.lstsq(A, b.reshape(-1), rcond=None)
    return w


def accel(v, w):
    speed = np.linalg.norm(v)
    return G - (KD / M) * speed * v + (KM / M) * np.cross(w, v)


def rollout(v0, w, t_rel, r0):
    r = np.array(r0, dtype=np.float64)
    v = np.array(v0, dtype=np.float64)
    out = [r.copy()]
    for i in range(1, len(t_rel)):
        dt = float(t_rel[i] - t_rel[i - 1])
        if not np.isfinite(dt) or dt <= 1e-9:
            dt = 1e-6
        k1v = accel(v, w)
        k1r = v
        k2v = accel(v + 0.5 * dt * k1v, w)
        k2r = v + 0.5 * dt * k1v
        k3v = accel(v + 0.5 * dt * k2v, w)
        k3r = v + 0.5 * dt * k2v
        k4v = accel(v + dt * k3v, w)
        k4r = v + dt * k3v
        v = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        r = r + dt / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
        out.append(r.copy())
    return np.array(out)


def fit_one(t, p):
    t_rel = t - t[0]
    ps, v, a = smooth_derivs(t, p)
    v0 = v[0]
    wg = linear_omega(v, a)
    r0 = p[0]

    def residual(x):
        return (rollout(x[3:6], x[:3], t_rel, r0) - p).ravel()

    lb = np.array([-1000.0, -1000.0, -1000.0, -25.0, -25.0, -25.0])
    ub = np.array([1000.0, 1000.0, 1000.0, 25.0, 25.0, 25.0])
    starts = [
        np.r_[wg, v0],
        np.r_[np.zeros(3), v0],
        np.r_[300.0 * np.array([0.0, 0.0, 1.0]), v0],
        np.r_[-300.0 * np.array([0.0, 0.0, 1.0]), v0],
        np.r_[300.0 * np.array([0.0, 1.0, 0.0]), v0],
        np.r_[300.0 * np.array([1.0, 0.0, 0.0]), v0],
    ]
    best = None
    for s in starts:
        s = np.clip(s, lb + 1e-8, ub - 1e-8)
        try:
            res = least_squares(residual, s, bounds=(lb, ub), max_nfev=100, x_scale="jac")
        except Exception:
            continue
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        return None
    w = best.x[:3]
    vv = best.x[3:6]
    rp = rollout(vv, w, t_rel, r0)
    rmse = float(np.sqrt(np.mean(np.sum((rp - p) ** 2, axis=1))))
    return {
        "omega": w,
        "v0": vv,
        "rmse": rmse,
        "cost": float(best.cost),
        "success": bool(best.success),
        "nfev": int(best.nfev),
    }


def angle_deg(pred, gt):
    pn = np.linalg.norm(pred)
    tn = np.linalg.norm(gt)
    if pn < 1e-12 or tn < 1e-12:
        return float("nan")
    c = np.clip(np.dot(pred, gt) / (pn * tn), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="trajectories")
    ap.add_argument("--index", default="trajectories/index.csv")
    ap.add_argument("--out", default="pretrain/output/fit_trajectories_physics")
    ap.add_argument("--min-points", type=int, default=30)
    ap.add_argument("--min-duration", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    def rel(p):
        q = Path(p)
        return q if q.is_absolute() else root / q
    out_dir = rel(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = select_usable(rel(args.dir), args.min_points, args.min_duration)
    if args.limit:
        items = items[: args.limit]
    index = load_index(rel(args.index))
    print(f"usable trajectories (geometry only): {len(items)}")

    rows = []
    for i, it in enumerate(items, 1):
        fit = fit_one(it["t"], it["p"])
        if fit is None:
            print(f"  skip {it['id']}: fit failed")
            continue
        w = fit["omega"]
        gt = index.get(it["id"])
        row = {
            "id": it["id"],
            "n": len(it["t"]),
            "duration": float(it["t"][-1] - it["t"][0]),
            "rmse_cm": fit["rmse"] * 100.0,
            "omega_x": float(w[0]),
            "omega_y": float(w[1]),
            "omega_z": float(w[2]),
            "omega_norm": float(np.linalg.norm(w)),
            "v0_x": float(fit["v0"][0]),
            "v0_y": float(fit["v0"][1]),
            "v0_z": float(fit["v0"][2]),
            "fit_success": int(fit["success"]),
            "nfev": fit["nfev"],
        }
        if gt is not None:
            row["gt_x"], row["gt_y"], row["gt_z"] = gt
            row["gt_norm"] = float(np.linalg.norm(gt))
            row["angle_deg"] = angle_deg(w, gt)
            row["mae"] = float(np.mean(np.abs(w - gt)))
            row["gt_outlier"] = int(np.linalg.norm(gt) > 628.0)
        else:
            row["gt_x"] = row["gt_y"] = row["gt_z"] = float("nan")
            row["gt_norm"] = float("nan")
            row["angle_deg"] = float("nan")
            row["mae"] = float("nan")
            row["gt_outlier"] = -1
        rows.append(row)
        if i % 10 == 0 or i == len(items):
            print(f"  {i}/{len(items)} id={it['id']} rmse={row['rmse_cm']:.2f}cm |w|={row['omega_norm']:.1f} angle={row['angle_deg']:.1f}")

    fields = ["id", "n", "duration", "rmse_cm", "omega_x", "omega_y", "omega_z", "omega_norm",
              "v0_x", "v0_y", "v0_z", "fit_success", "nfev",
              "gt_x", "gt_y", "gt_z", "gt_norm", "angle_deg", "mae", "gt_outlier"]
    with open(out_dir / "fit_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    valid = [r for r in rows if np.isfinite(r["angle_deg"])]
    inliers = [r for r in valid if r["gt_outlier"] == 0]
    summary = {
        "n_usable": len(items),
        "n_fitted": len(rows),
        "n_with_gt": len(valid),
        "n_gt_outliers": int(sum(1 for r in valid if r["gt_outlier"] == 1)),
        "rmse_cm_median": float(np.median([r["rmse_cm"] for r in rows])),
        "rmse_cm_p95": float(np.percentile([r["rmse_cm"] for r in rows], 95)),
        "rmse_cm_max": float(np.max([r["rmse_cm"] for r in rows])),
        "fit_success_rate": float(np.mean([r["fit_success"] for r in rows])),
        "all_gt": {
            "mae_mean": float(np.mean([r["mae"] for r in valid])) if valid else float("nan"),
            "angle_mean": float(np.mean([r["angle_deg"] for r in valid])) if valid else float("nan"),
            "angle_median": float(np.median([r["angle_deg"] for r in valid])) if valid else float("nan"),
        },
        "inlier_gt": {
            "n": len(inliers),
            "mae_mean": float(np.mean([r["mae"] for r in inliers])) if inliers else float("nan"),
            "angle_mean": float(np.mean([r["angle_deg"] for r in inliers])) if inliers else float("nan"),
            "angle_median": float(np.median([r["angle_deg"] for r in inliers])) if inliers else float("nan"),
        },
        "per_trajectory": rows,
    }
    with open(out_dir / "fit_results.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("\nsummary")
    print(f"  fitted: {len(rows)} | with GT: {len(valid)} | GT outliers: {summary['n_gt_outliers']}")
    print(f"  fit position RMSE: median {summary['rmse_cm_median']:.2f} cm | p95 {summary['rmse_cm_p95']:.2f} | max {summary['rmse_cm_max']:.2f}")
    print(f"  fit success rate: {summary['fit_success_rate']*100:.1f}%")
    print(f"  vs index.csv (all):   MAE {summary['all_gt']['mae_mean']:.1f} | angle {summary['all_gt']['angle_mean']:.1f}")
    print(f"  vs index.csv (inlier): MAE {summary['inlier_gt']['mae_mean']:.1f} | angle {summary['inlier_gt']['angle_mean']:.1f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        flat = axes.ravel()
        if inliers:
            for k, ax_name in enumerate("xyz"):
                g = np.array([r[f"gt_{ax_name}"] for r in inliers])
                f = np.array([r[f"omega_{ax_name}"] for r in inliers])
                flat[k].scatter(g, f, s=12, alpha=0.7)
                lo = min(g.min(), f.min())
                hi = max(g.max(), f.max())
                flat[k].plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
                flat[k].set_xlabel(f"index {ax_name}")
                flat[k].set_ylabel(f"fit {ax_name}")
                flat[k].set_title(f"{ax_name} component")
            flat[3].hist([r["angle_deg"] for r in inliers], bins=20, alpha=0.8)
            flat[3].set_title("angle error (deg), inliers")
        fig.tight_layout()
        fig.savefig(out_dir / "fit_vs_index.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"plot skipped: {exc}")

    print(f"\nsaved: {out_dir}")


if __name__ == "__main__":
    main()
