import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import splprep, splev

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import SpinCNN
from fit_compare import load_clean, load_meta, poly_fit, model_fit, GRID


def smooth_xy(xy, n=300):
    xy = np.asarray(xy, dtype=np.float64)
    keep = np.ones(len(xy), dtype=bool)
    if len(xy) > 1:
        keep[1:] = np.any(np.abs(np.diff(xy, axis=0)) > 1e-9, axis=1)
    xy = xy[keep]
    if len(xy) < 4:
        return xy[:, 0], xy[:, 1]
    tck, _ = splprep([xy[:, 0], xy[:, 1]], s=0, k=3)
    u = np.linspace(0.0, 1.0, n)
    x, y = splev(u, tck)
    return np.asarray(x), np.asarray(y)


def rmse_cm(pred, true):
    return float(np.sqrt(np.mean(np.sum((pred - true) ** 2, axis=1))) * 100.0)


def draw(ax, p_true, p_quad, p_cubic, p_model, tid, title_prefix="3D_gt", legend=True):
    x, y = smooth_xy(p_true[:, :2])
    ax.plot(x, y, "-", color="black", linewidth=2.2, label="true")
    x, y = smooth_xy(p_quad[:, :2])
    ax.plot(x, y, "--", color="#1f77b4", linewidth=1.5, label=f"quadratic ({rmse_cm(p_quad, p_true):.2f} cm)")
    x, y = smooth_xy(p_cubic[:, :2])
    ax.plot(x, y, "--", color="#ff7f0e", linewidth=1.5, label=f"cubic ({rmse_cm(p_cubic, p_true):.2f} cm)")
    x, y = smooth_xy(p_model[:, :2])
    ax.plot(x, y, ":", color="#2ca02c", linewidth=2.0, label=f"model ({rmse_cm(p_model, p_true):.2f} cm)")

    ax.scatter(p_true[0, 0], p_true[0, 1], s=40, color="black", zorder=5)
    ax.scatter(p_true[-1, 0], p_true[-1, 1], s=40, marker="s", color="black", zorder=5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"{title_prefix} {tid:03d} — XY projection")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)
    if legend:
        ax.legend(fontsize=7, loc="best")


def draw_table(ax):
    from matplotlib.patches import Rectangle
    rect = Rectangle((-0.7625, -1.37), 1.525, 2.74, fill=False, linestyle="--", linewidth=0.8, color="gray", alpha=0.5)
    ax.add_patch(rect)
    ax.plot([-0.7625, 0.7625], [0, 0], linestyle="--", linewidth=0.8, color="gray", alpha=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default="prep_3d_gt/output/3D_gt_clean")
    ap.add_argument("--meta", default="prep_3d_gt/output/3D_gt_meta.csv")
    ap.add_argument("--checkpoint", default="pretrain/output/finetune/finetuned_model.pt")
    ap.add_argument("--out", default="pretrain/output/fit_compare/xy")
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

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    for tid in ids:
        t, p = load_clean(clean_dir / f"{tid:03d}.csv")
        T = t[-1] - t[0]
        p_true = np.column_stack([np.interp(GRID, (t - t[0]) / max(T, 1e-12), p[:, j]) for j in range(3)])
        p_quad = poly_fit(t, p, GRID, 2)
        p_cubic = poly_fit(t, p, GRID, 3)
        r = meta[tid]
        v0_world = np.array([float(r["v0_x"]), float(r["v0_y"]), float(r["v0_z"])])
        p_model, w_world, theta = model_fit(model, norm, t, p, v0_world)

        fig, ax = plt.subplots(figsize=(7, 8))
        draw_table(ax)
        draw(ax, p_true, p_quad, p_cubic, p_model, tid)
        fig.tight_layout()
        path = out_dir / f"xy_{tid:03d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"saved {path}")

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    for ax, tid in zip(axes.ravel(), ids):
        t, p = load_clean(clean_dir / f"{tid:03d}.csv")
        T = t[-1] - t[0]
        p_true = np.column_stack([np.interp(GRID, (t - t[0]) / max(T, 1e-12), p[:, j]) for j in range(3)])
        p_quad = poly_fit(t, p, GRID, 2)
        p_cubic = poly_fit(t, p, GRID, 3)
        r = meta[tid]
        v0_world = np.array([float(r["v0_x"]), float(r["v0_y"]), float(r["v0_z"])])
        p_model, _, _ = model_fit(model, norm, t, p, v0_world)
        draw_table(ax)
        draw(ax, p_true, p_quad, p_cubic, p_model, tid, title_prefix="id", legend=False)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    fig.tight_layout()
    all_path = out_dir / "xy_all.png"
    fig.savefig(all_path, dpi=150)
    plt.close(fig)
    print(f"saved {all_path}")


if __name__ == "__main__":
    main()
