import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import SpinCNN
from finetune import rotate_xy, rotate_back_np, KD, KM
from plot_xy import smooth_xy, draw, draw_table

G = np.array([0.0, 0.0, -9.81])
R = 0.02
M = 2.7e-3
L = 64
GRID = np.linspace(0.0, 1.0, L)


def load_raw(path):
    arr = np.loadtxt(str(path), delimiter=",", skiprows=1)
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


def load_meta(path):
    meta = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            meta[int(r["id"])] = r
    return meta


def poly_fit_raw(t, p, deg):
    tn = (t - t[0]) / max(t[-1] - t[0], 1e-12)
    pred = np.zeros_like(p)
    for j in range(3):
        coef = np.polyfit(tn, p[:, j], deg)
        pred[:, j] = np.polyval(coef, tn)
    return pred


def accel(v, w):
    speed = np.linalg.norm(v)
    return G - (KD / M) * speed * v + (KM / M) * np.cross(w, v)


def rollout_at_times(v0, w, t_rel):
    r = np.zeros(3)
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


def predict_spin(model, norm, t, p):
    T = t[-1] - t[0]
    v = np.gradient(p, t, axis=0)
    theta = float(np.arctan2(v[0, 0], v[0, 1]))
    p_c = rotate_xy(p - p[0], theta)
    v_c = rotate_xy(v, theta)
    tau = (t - t[0]) / max(T, 1e-12)
    p_r = np.column_stack([np.interp(GRID, tau, p_c[:, j]) for j in range(3)])
    v_r = np.column_stack([np.interp(GRID, tau, v_c[:, j]) for j in range(3)])
    X = np.zeros((1, 7, L), dtype=np.float32)
    X[0, :3] = p_r.T
    X[0, 3:6] = v_r.T
    X[0, 6] = GRID
    x_mean = np.array(norm["x_mean"], dtype=np.float32).reshape(1, 7, 1)
    x_std = np.array(norm["x_std"], dtype=np.float32).reshape(1, 7, 1)
    X = (X - x_mean) / x_std
    y_mean = np.array(norm["y_mean"], dtype=np.float32)
    y_std = np.array(norm["y_std"], dtype=np.float32)
    with torch.no_grad():
        pred_n = model(torch.from_numpy(X)).numpy()[0]
    return pred_n * y_std + y_mean, theta


def model_fit_raw(model, norm, t, p, v0_world):
    w_c, theta = predict_spin(model, norm, t, p)
    v0_c = rotate_xy(v0_world[None, :], theta)[0]
    t_rel = t - t[0]
    rp_c = rollout_at_times(v0_c, w_c, t_rel)
    theta_arr = np.full(len(t_rel), theta)
    p_world = rotate_back_np(rp_c, theta_arr) + p[0]
    w_world = rotate_back_np(w_c[None, :], np.array([theta]))[0]
    return p_world, w_world


def rmse_cm(pred, true):
    return float(np.sqrt(np.mean(np.sum((pred - true) ** 2, axis=1))) * 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="3D_gt")
    ap.add_argument("--meta", default="prep_3d_gt/output/3D_gt_meta.csv")
    ap.add_argument("--checkpoint", default="pretrain/output/finetune/finetuned_model.pt")
    ap.add_argument("--prev", default="pretrain/output/fit_compare/fit_compare.csv")
    ap.add_argument("--out", default="pretrain/output/fit_compare_raw")
    ap.add_argument("--ids", default="1,2,3,4,5,6,7,8,9,10")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    def rel(p):
        q = Path(p)
        return q if q.is_absolute() else root / q
    raw_dir = rel(args.raw_dir)
    out_dir = rel(args.out)
    xy_dir = out_dir / "xy"
    xy_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(rel(args.checkpoint), map_location="cpu", weights_only=False)
    model = SpinCNN(in_ch=7, out=3)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    norm = ckpt["norm"]
    meta = load_meta(rel(args.meta))
    ids = [int(x) for x in args.ids.split(",")]

    prev = {}
    prev_path = rel(args.prev)
    if prev_path.exists():
        with open(prev_path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                prev[int(r["id"])] = r

    rows = []
    curves = {}
    for tid in ids:
        t_raw, p_raw = load_raw(raw_dir / f"{tid:03d}.csv")
        bi = first_bounce(t_raw, p_raw)
        if bi is not None and bi >= 4:
            t_use = t_raw[:bi]
            p_use = p_raw[:bi]
            bounce = f"bounce@{bi}"
        else:
            t_use = t_raw
            p_use = p_raw
            bounce = "no_bounce"
        T = t_use[-1] - t_use[0]
        p_quad = poly_fit_raw(t_use, p_use, 2)
        p_cubic = poly_fit_raw(t_use, p_use, 3)
        r = meta[tid]
        v0_world = np.array([float(r["v0_x"]), float(r["v0_y"]), float(r["v0_z"])])
        p_model, w_world = model_fit_raw(model, norm, t_use, p_use, v0_world)

        row = {
            "id": tid,
            "n_raw": len(t_use),
            "duration": float(T),
            "bounce": bounce,
            "rmse_quad_raw_cm": rmse_cm(p_quad, p_use),
            "rmse_cubic_raw_cm": rmse_cm(p_cubic, p_use),
            "rmse_model_raw_cm": rmse_cm(p_model, p_use),
            "omega_x": float(w_world[0]),
            "omega_y": float(w_world[1]),
            "omega_z": float(w_world[2]),
            "omega_norm": float(np.linalg.norm(w_world)),
        }
        if tid in prev:
            row["rmse_quad_resampled_cm"] = float(prev[tid]["rmse_quad_cm"])
            row["rmse_cubic_resampled_cm"] = float(prev[tid]["rmse_cubic_cm"])
            row["rmse_model_resampled_cm"] = float(prev[tid]["rmse_model_cm"])
        else:
            row["rmse_quad_resampled_cm"] = float("nan")
            row["rmse_cubic_resampled_cm"] = float("nan")
            row["rmse_model_resampled_cm"] = float("nan")
        rows.append(row)
        curves[tid] = {"t": t_use - t_use[0], "true": p_use, "quad": p_quad, "cubic": p_cubic, "model": p_model}

    fields = ["id", "n_raw", "duration", "bounce",
              "rmse_quad_raw_cm", "rmse_cubic_raw_cm", "rmse_model_raw_cm",
              "rmse_quad_resampled_cm", "rmse_cubic_resampled_cm", "rmse_model_resampled_cm",
              "omega_x", "omega_y", "omega_z", "omega_norm"]
    with open(out_dir / "fit_compare_raw.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    def agg(key):
        v = np.array([r[key] for r in rows if np.isfinite(r[key])])
        if len(v) == 0:
            return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
        return {"mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}

    summary = {
        "ids": ids,
        "raw": {
            "quad": agg("rmse_quad_raw_cm"),
            "cubic": agg("rmse_cubic_raw_cm"),
            "model": agg("rmse_model_raw_cm"),
        },
        "resampled": {
            "quad": agg("rmse_quad_resampled_cm"),
            "cubic": agg("rmse_cubic_resampled_cm"),
            "model": agg("rmse_model_resampled_cm"),
        },
        "per_trajectory": rows,
    }
    with open(out_dir / "fit_compare_raw.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"{'id':>3} {'n_raw':>5} {'dur':>5} {'quad_raw':>9} {'cubic_raw':>9} {'model_raw':>9} "
          f"{'quad_rs':>8} {'cubic_rs':>8} {'model_rs':>8}")
    for r in rows:
        print(f"{r['id']:>3} {r['n_raw']:>5} {r['duration']:>5.3f} "
              f"{r['rmse_quad_raw_cm']:>9.2f} {r['rmse_cubic_raw_cm']:>9.2f} {r['rmse_model_raw_cm']:>9.2f} "
              f"{r['rmse_quad_resampled_cm']:>8.2f} {r['rmse_cubic_resampled_cm']:>8.2f} {r['rmse_model_resampled_cm']:>8.2f}")
    print("\nsummary RMSE (cm), raw points:")
    for name in ("quad", "cubic", "model"):
        s = summary["raw"][name]
        print(f"  {name}: mean {s['mean']:.2f} | median {s['median']:.2f} | p95 {s['p95']:.2f} | max {s['max']:.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.arange(len(rows))
        wbar = 0.25
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].bar(x - wbar, [r["rmse_quad_raw_cm"] for r in rows], width=wbar, label="quadratic")
        axes[0].bar(x, [r["rmse_cubic_raw_cm"] for r in rows], width=wbar, label="cubic")
        axes[0].bar(x + wbar, [r["rmse_model_raw_cm"] for r in rows], width=wbar, label="model")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([str(r["id"]) for r in rows])
        axes[0].set_ylabel("RMSE (cm)")
        axes[0].set_title("raw-point RMSE")
        axes[0].legend()

        tid = rows[0]["id"]
        c = curves[tid]
        axes[1].plot(c["t"], c["true"][:, 2], "k-", label="true", linewidth=2)
        axes[1].plot(c["t"], c["quad"][:, 2], "--", label="quadratic")
        axes[1].plot(c["t"], c["cubic"][:, 2], "--", label="cubic")
        axes[1].plot(c["t"], c["model"][:, 2], ":", label="model")
        axes[1].set_title(f"example Z trajectory (id {tid})")
        axes[1].set_xlabel("t (s)")
        axes[1].set_ylabel("z (m)")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(out_dir / "fit_compare_raw.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        methods = ["quadratic", "cubic", "model"]
        raw_means = [summary["raw"]["quad"]["mean"], summary["raw"]["cubic"]["mean"], summary["raw"]["model"]["mean"]]
        rs_means = [summary["resampled"]["quad"]["mean"], summary["resampled"]["cubic"]["mean"], summary["resampled"]["model"]["mean"]]
        xm = np.arange(3)
        ax.bar(xm - 0.18, raw_means, width=0.36, label="raw points")
        ax.bar(xm + 0.18, rs_means, width=0.36, label="resampled 64-grid")
        ax.set_xticks(xm)
        ax.set_xticklabels(methods)
        ax.set_ylabel("mean RMSE (cm)")
        ax.set_title("raw vs resampled evaluation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "compare_raw_vs_resampled.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(2, 5, figsize=(22, 9))
        for ax, tid in zip(axes.ravel(), ids):
            c = curves[tid]
            draw_table(ax)
            draw(ax, c["true"], c["quad"], c["cubic"], c["model"], tid, title_prefix="3D_gt raw", legend=False)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        fig.tight_layout()
        fig.savefig(xy_dir / "xy_all.png", dpi=150)
        plt.close(fig)

        for tid in ids:
            c = curves[tid]
            fig, ax = plt.subplots(figsize=(7, 8))
            draw_table(ax)
            draw(ax, c["true"], c["quad"], c["cubic"], c["model"], tid, title_prefix="3D_gt raw")
            fig.tight_layout()
            fig.savefig(xy_dir / f"xy_{tid:03d}.png", dpi=150)
            plt.close(fig)
        print(f"xy figures saved in {xy_dir}")
    except Exception as exc:
        print(f"plot skipped: {exc}")

    print(f"\nsaved: {out_dir}")


if __name__ == "__main__":
    main()
