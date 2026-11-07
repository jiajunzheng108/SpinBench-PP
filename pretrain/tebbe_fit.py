import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_trajectories_physics import (
    read_traj,
    first_bounce,
    select_usable,
    load_index,
    rollout,
    G,
    M,
    KD,
    KM,
)

KD_M = KD / M
KM_M = KM / M


def cubic_smooth(t, p):
    t0 = t[0]
    T = t[-1] - t[0]
    tn = (t - t0) / max(T, 1e-12)
    mask = np.ones(len(t), dtype=bool)
    for _ in range(3):
        coefs = [np.polyfit(tn[mask], p[mask, j], 3) for j in range(3)]
        pred = np.column_stack([np.polyval(coefs[j], tn) for j in range(3)])
        resid = np.linalg.norm(pred - p, axis=1)
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med))) + 1e-9
        new_mask = resid <= med + 4.0 * 1.4826 * mad
        if new_mask.sum() < max(10, int(0.5 * len(t))):
            break
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    coefs = [np.polyfit(tn, p[:, j], 3) for j in range(3)]
    v = np.column_stack([np.polyval(np.polyder(coefs[j], 1), tn) / max(T, 1e-12) for j in range(3)])
    a = np.column_stack([np.polyval(np.polyder(coefs[j], 2), tn) / max(T, 1e-12) ** 2 for j in range(3)])
    return v, a, mask, tn


def build_system(v, a):
    rows = []
    rhs = []
    for i in range(len(v)):
        vx, vy, vz = v[i]
        rows.append([[0.0, vz, -vy], [-vz, 0.0, vx], [vy, -vx, 0.0]])
        rhs.append((a[i] - G + KD_M * np.linalg.norm(v[i]) * v[i]) / KM_M)
    return np.concatenate([np.array(r, dtype=np.float64) for r in rows], axis=0), np.concatenate(rhs, axis=0)


def solve_svd(mat, rhs, rcond=1e-6):
    u, s, vt = np.linalg.svd(mat, full_matrices=False)
    s0 = float(s[0]) if len(s) else 0.0
    ratio = float(s[-1] / s0) if s0 > 0 else 0.0
    cond = float(s0 / s[-1]) if len(s) and s[-1] > 0 else float("inf")
    tol = s0 * rcond
    sinv = np.where(s > tol, 1.0 / s, 0.0)
    w = vt.T @ (sinv * (u.T @ rhs))
    return w, s, ratio, cond


def perp_component(w, v_hat):
    return w - np.dot(w, v_hat) * v_hat


def angle_deg(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    c = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def synthetic_check():
    t = np.arange(0.0, 0.4, 0.0056)
    r0 = np.array([0.0, 1.5, 0.3])
    v0 = np.array([0.2, -8.0, 1.5])
    w_true = np.array([300.0, -50.0, -100.0])
    p = rollout(v0, w_true, t, r0)
    v, a, mask, tn = cubic_smooth(t, p)
    mat, rhs = build_system(v, a)
    w_fit, s, ratio, cond = solve_svd(mat, rhs)
    v_hat = v.mean(axis=0)
    v_hat /= np.linalg.norm(v_hat)
    w_true_perp = perp_component(w_true, v_hat)
    w_fit_perp = perp_component(w_fit, v_hat)
    print("synthetic check")
    print("  true w:", np.round(w_true, 2))
    print("  fit  w:", np.round(w_fit, 2))
    print("  sigma3/sigma1:", f"{ratio:.3e}", "cond:", f"{cond:.1f}")
    print("  full angle:", f"{angle_deg(w_fit, w_true):.2f}")
    print("  perp angle:", f"{angle_deg(w_fit_perp, w_true_perp):.2f}")
    print("  perp mae:", f"{np.mean(np.abs(w_fit_perp - w_true_perp)):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="trajectories")
    ap.add_argument("--index", default="trajectories/index.csv")
    ap.add_argument("--out", default="pretrain/output/tebbe_fit")
    ap.add_argument("--min-points", type=int, default=30)
    ap.add_argument("--min-duration", type=float, default=0.25)
    ap.add_argument("--rcond", type=float, default=1e-6)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    if args.synthetic:
        synthetic_check()
        return

    root = Path(__file__).resolve().parent.parent
    def rel(p):
        q = Path(p)
        return q if q.is_absolute() else root / q
    out_dir = rel(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = select_usable(rel(args.dir), args.min_points, args.min_duration)
    index = load_index(rel(args.index))
    print(f"usable trajectories: {len(items)}")

    rows = []
    for i, it in enumerate(items, 1):
        t, p = it["t"], it["p"]
        v, a, mask, tn = cubic_smooth(t, p)
        mat, rhs = build_system(v, a)
        w, s, ratio, cond = solve_svd(mat, rhs, args.rcond)
        t_rel = t - t[0]
        rp = rollout(v[0], w, t_rel, p[0])
        rmse = float(np.sqrt(np.mean(np.sum((rp - p) ** 2, axis=1))) * 100.0)
        v_hat = v[mask].mean(axis=0)
        if np.linalg.norm(v_hat) > 1e-12:
            v_hat = v_hat / np.linalg.norm(v_hat)
        gt = index.get(it["id"])
        row = {
            "id": it["id"],
            "n": len(t),
            "duration": float(t[-1] - t[0]),
            "n_inliers": int(mask.sum()),
            "rmse_cm": rmse,
            "sigma3_sigma1": ratio,
            "cond": cond,
            "omega_x": float(w[0]),
            "omega_y": float(w[1]),
            "omega_z": float(w[2]),
            "omega_norm": float(np.linalg.norm(w)),
        }
        if gt is not None:
            gt_perp = perp_component(gt, v_hat)
            fit_perp = perp_component(w, v_hat)
            row.update({
                "gt_x": float(gt[0]), "gt_y": float(gt[1]), "gt_z": float(gt[2]),
                "gt_norm": float(np.linalg.norm(gt)),
                "gt_outlier": int(np.linalg.norm(gt) > 628.0),
                "full_angle": angle_deg(w, gt),
                "full_mae": float(np.mean(np.abs(w - gt))),
                "fit_perp_x": float(fit_perp[0]), "fit_perp_y": float(fit_perp[1]), "fit_perp_z": float(fit_perp[2]),
                "gt_perp_x": float(gt_perp[0]), "gt_perp_y": float(gt_perp[1]), "gt_perp_z": float(gt_perp[2]),
                "perp_angle": angle_deg(fit_perp, gt_perp),
                "perp_mae": float(np.mean(np.abs(fit_perp - gt_perp))),
            })
        else:
            for k in ("gt_x", "gt_y", "gt_z", "gt_norm", "gt_outlier", "full_angle", "full_mae",
                      "fit_perp_x", "fit_perp_y", "fit_perp_z", "gt_perp_x", "gt_perp_y", "gt_perp_z",
                      "perp_angle", "perp_mae"):
                row[k] = float("nan")
        rows.append(row)
        if i % 10 == 0 or i == len(items):
            print(f"  {i}/{len(items)} id={it['id']} rmse={rmse:.2f}cm ratio={ratio:.2e} full_angle={row['full_angle']:.1f} perp_angle={row['perp_angle']:.1f}")

    fields = ["id", "n", "duration", "n_inliers", "rmse_cm", "sigma3_sigma1", "cond",
              "omega_x", "omega_y", "omega_z", "omega_norm",
              "gt_x", "gt_y", "gt_z", "gt_norm", "gt_outlier",
              "full_angle", "full_mae",
              "fit_perp_x", "fit_perp_y", "fit_perp_z",
              "gt_perp_x", "gt_perp_y", "gt_perp_z",
              "perp_angle", "perp_mae"]
    with open(out_dir / "tebbe_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    valid = [r for r in rows if np.isfinite(r["full_angle"])]
    inl = [r for r in valid if r["gt_outlier"] == 0]
    groups = {
        "good (ratio>1e-2)": [r for r in inl if r["sigma3_sigma1"] > 1e-2],
        "medium (1e-3~1e-2)": [r for r in inl if 1e-3 < r["sigma3_sigma1"] <= 1e-2],
        "poor (<1e-3)": [r for r in inl if r["sigma3_sigma1"] <= 1e-3],
    }

    def agg(sub, key):
        if not sub:
            return float("nan")
        return float(np.mean([r[key] for r in sub]))

    summary = {
        "n_usable": len(items),
        "n_valid": len(valid),
        "n_gt_outliers": int(sum(1 for r in valid if r["gt_outlier"] == 1)),
        "rmse_cm_median": float(np.median([r["rmse_cm"] for r in rows])),
        "rmse_cm_p95": float(np.percentile([r["rmse_cm"] for r in rows], 95)),
        "ratio_median": float(np.median([r["sigma3_sigma1"] for r in rows])),
        "ratio_p95": float(np.percentile([r["sigma3_sigma1"] for r in rows], 95)),
        "count_ratio_gt_1e-2": int(sum(1 for r in rows if r["sigma3_sigma1"] > 1e-2)),
        "count_ratio_lt_1e-3": int(sum(1 for r in rows if r["sigma3_sigma1"] < 1e-3)),
        "all_inliers": {
            "n": len(inl),
            "full_mae": agg(inl, "full_mae"),
            "full_angle": agg(inl, "full_angle"),
            "perp_mae": agg(inl, "perp_mae"),
            "perp_angle": agg(inl, "perp_angle"),
        },
        "groups": {},
    }
    for name, sub in groups.items():
        summary["groups"][name] = {
            "n": len(sub),
            "full_mae": agg(sub, "full_mae"),
            "full_angle": agg(sub, "full_angle"),
            "perp_mae": agg(sub, "perp_mae"),
            "perp_angle": agg(sub, "perp_angle"),
            "rmse_cm": agg(sub, "rmse_cm"),
        }
    with open(out_dir / "tebbe_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("\nsummary")
    print(f"  fit RMSE median/p95: {summary['rmse_cm_median']:.2f} / {summary['rmse_cm_p95']:.2f} cm")
    print(f"  sigma3/sigma1 median/p95: {summary['ratio_median']:.3e} / {summary['ratio_p95']:.3e}")
    print(f"  ratio>1e-2: {summary['count_ratio_gt_1e-2']} | ratio<1e-3: {summary['count_ratio_lt_1e-3']}")
    print(f"  inliers: full MAE {summary['all_inliers']['full_mae']:.1f} / angle {summary['all_inliers']['full_angle']:.1f} | "
          f"perp MAE {summary['all_inliers']['perp_mae']:.1f} / angle {summary['all_inliers']['perp_angle']:.1f}")
    for name, g in summary["groups"].items():
        print(f"  {name}: n={g['n']} full {g['full_angle']:.1f} deg | perp {g['perp_angle']:.1f} deg | rmse {g['rmse_cm']:.2f} cm")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        ratios = np.array([r["sigma3_sigma1"] for r in inl])
        axes[0].hist(np.log10(np.maximum(ratios, 1e-12)), bins=20, alpha=0.8)
        axes[0].set_xlabel("log10(sigma3/sigma1)")
        axes[0].set_title("identifiability")
        axes[1].scatter(ratios, [r["full_angle"] for r in inl], s=12, alpha=0.7, label="full")
        axes[1].scatter(ratios, [r["perp_angle"] for r in inl], s=12, alpha=0.7, label="perp")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("sigma3/sigma1")
        axes[1].set_ylabel("angle error (deg)")
        axes[1].set_title("angle vs identifiability")
        axes[1].legend()
        gx = np.array([r["gt_perp_x"] for r in inl])
        fx = np.array([r["fit_perp_x"] for r in inl])
        axes[2].scatter(gx, fx, s=12, alpha=0.7)
        lo = min(gx.min(), fx.min())
        hi = max(gx.max(), fx.max())
        axes[2].plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
        axes[2].set_xlabel("index perp x")
        axes[2].set_ylabel("fit perp x")
        axes[2].set_title("perpendicular component")
        fig.tight_layout()
        fig.savefig(out_dir / "tebbe_analysis.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"plot skipped: {exc}")

    print(f"\nsaved: {out_dir}")


if __name__ == "__main__":
    main()
