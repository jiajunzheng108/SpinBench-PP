import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import SpinCNN
from finetune import rollout, rotate_xy, rotate_back_np, KD, KM

L = 64
GRID = np.linspace(0.0, 1.0, L)


def load_clean(path):
    arr = np.loadtxt(str(path), delimiter=",", skiprows=1)
    return arr[:, 0].astype(np.float64), arr[:, 1:4].astype(np.float64)


def load_meta(path):
    meta = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            meta[int(r["id"])] = r
    return meta


def poly_fit(t, p, grid, deg):
    tn = (t - t[0]) / max(t[-1] - t[0], 1e-12)
    pred = np.zeros((len(grid), 3))
    for j in range(3):
        coef = np.polyfit(tn, p[:, j], deg)
        pred[:, j] = np.polyval(coef, grid)
    return pred


def model_fit(model, norm, t, p, v0_world):
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
    w_c = pred_n * y_std + y_mean
    v0_c = rotate_xy(v0_world[None, :], theta)[0]
    dts = torch.tensor([T / (L - 1)], dtype=torch.float32)
    w_t = torch.tensor(w_c, dtype=torch.float32).unsqueeze(0)
    v0_t = torch.tensor(v0_c, dtype=torch.float32).unsqueeze(0)
    rp_c = rollout(v0_t, w_t, dts, L, KD, KM).numpy()[0]
    theta_arr = np.full(L, theta)
    p_model = rotate_back_np(rp_c, theta_arr) + p[0]
    w_world = rotate_back_np(w_c[None, :], np.array([theta]))[0]
    return p_model, w_world, theta


def rmse(pred, true):
    return float(np.sqrt(np.mean(np.sum((pred - true) ** 2, axis=1))) * 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default="prep_3d_gt/output/3D_gt_clean")
    ap.add_argument("--meta", default="prep_3d_gt/output/3D_gt_meta.csv")
    ap.add_argument("--checkpoint", default="pretrain/output/finetune/finetuned_model.pt")
    ap.add_argument("--out", default="pretrain/output/fit_compare")
    ap.add_argument("--ids", default="1,2,3,4,5,6,7,8,9,10")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    def rel(p):
        q = Path(p)
        return q if q.is_absolute() else root / q
    clean_dir = rel(args.clean_dir)
    out_dir = rel(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(rel(args.checkpoint), map_location="cpu", weights_only=False)
    model = SpinCNN(in_ch=7, out=3)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    norm = ckpt["norm"]
    meta = load_meta(rel(args.meta))
    ids = [int(x) for x in args.ids.split(",")]

    rows = []
    preds = {}
    trues = {}
    for tid in ids:
        t, p = load_clean(clean_dir / f"{tid:03d}.csv")
        T = t[-1] - t[0]
        t_grid = GRID * T
        p_true = np.column_stack([np.interp(GRID, (t - t[0]) / max(T, 1e-12), p[:, j]) for j in range(3)])

        p_quad = poly_fit(t, p, GRID, 2)
        p_cubic = poly_fit(t, p, GRID, 3)

        r = meta[tid]
        v0_world = np.array([float(r["v0_x"]), float(r["v0_y"]), float(r["v0_z"])])
        p_model, w_world, theta = model_fit(model, norm, t, p, v0_world)

        row = {
            "id": tid,
            "n_points": len(t),
            "duration": float(T),
            "rmse_quad_cm": rmse(p_quad, p_true),
            "rmse_cubic_cm": rmse(p_cubic, p_true),
            "rmse_model_cm": rmse(p_model, p_true),
            "omega_x": float(w_world[0]),
            "omega_y": float(w_world[1]),
            "omega_z": float(w_world[2]),
            "omega_norm": float(np.linalg.norm(w_world)),
            "theta": theta,
        }
        rows.append(row)
        preds[tid] = {"true": p_true, "quad": p_quad, "cubic": p_cubic, "model": p_model}
        trues[tid] = t_grid

    fields = ["id", "n_points", "duration", "rmse_quad_cm", "rmse_cubic_cm", "rmse_model_cm",
              "omega_x", "omega_y", "omega_z", "omega_norm", "theta"]
    with open(out_dir / "fit_compare.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    def agg(key):
        v = np.array([r[key] for r in rows])
        return {"mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}

    summary = {
        "ids": ids,
        "rmse_quad_cm": agg("rmse_quad_cm"),
        "rmse_cubic_cm": agg("rmse_cubic_cm"),
        "rmse_model_cm": agg("rmse_model_cm"),
        "per_trajectory": rows,
    }
    with open(out_dir / "fit_compare.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"{'id':>3} {'n':>4} {'dur':>5} {'quad':>8} {'cubic':>8} {'model':>8}  {'wx':>8} {'wy':>8} {'wz':>8}")
    for r in rows:
        print(f"{r['id']:>3} {r['n_points']:>4} {r['duration']:>5.3f} "
              f"{r['rmse_quad_cm']:>8.2f} {r['rmse_cubic_cm']:>8.2f} {r['rmse_model_cm']:>8.2f}  "
              f"{r['omega_x']:>8.1f} {r['omega_y']:>8.1f} {r['omega_z']:>8.1f}")
    print("\nsummary RMSE (cm):")
    for key in ("rmse_quad_cm", "rmse_cubic_cm", "rmse_model_cm"):
        s = summary[key]
        print(f"  {key}: mean {s['mean']:.2f} | median {s['median']:.2f} | p95 {s['p95']:.2f} | max {s['max']:.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        x = np.arange(len(rows))
        w = 0.25
        axes[0].bar(x - w, [r["rmse_quad_cm"] for r in rows], width=w, label="quadratic")
        axes[0].bar(x, [r["rmse_cubic_cm"] for r in rows], width=w, label="cubic")
        axes[0].bar(x + w, [r["rmse_model_cm"] for r in rows], width=w, label="model")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([str(r["id"]) for r in rows])
        axes[0].set_ylabel("RMSE (cm)")
        axes[0].set_title("per-trajectory RMSE")
        axes[0].legend()

        tid = rows[0]["id"]
        tg = trues[tid]
        d = preds[tid]
        axes[1].plot(tg, d["true"][:, 2], "k-", label="true", linewidth=2)
        axes[1].plot(tg, d["quad"][:, 2], "--", label="quadratic")
        axes[1].plot(tg, d["cubic"][:, 2], "--", label="cubic")
        axes[1].plot(tg, d["model"][:, 2], ":", label="model")
        axes[1].set_title(f"example Z trajectory (id {tid})")
        axes[1].set_xlabel("t (s)")
        axes[1].set_ylabel("z (m)")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(out_dir / "fit_compare.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"plot skipped: {exc}")

    print(f"\nsaved: {out_dir}")


if __name__ == "__main__":
    main()
