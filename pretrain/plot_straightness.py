import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import splprep, splev

sys.path.insert(0, str(Path(__file__).resolve().parent))


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


def load_spin_index(path):
    spins = {}
    with open(path, newline="", encoding="utf-8") as fh:
        r = csv.reader(fh)
        next(r)
        for row in r:
            if len(row) < 4:
                continue
            spins[int(row[0])] = np.array([float(row[1]), float(row[2]), float(row[3])])
    return spins


def smooth_xy(xy, n=300):
    xy = np.asarray(xy, dtype=np.float64)
    keep = np.ones(len(xy), dtype=bool)
    if len(xy) > 1:
        keep[1:] = np.any(np.abs(np.diff(xy, axis=0)) > 1e-9, axis=1)
    xy = xy[keep]
    if len(xy) < 4:
        return xy[:, 0], xy[:, 1]
    k = min(3, len(xy) - 1)
    tck, _ = splprep([xy[:, 0], xy[:, 1]], s=0, k=k)
    u = np.linspace(0.0, 1.0, n)
    x, y = splev(u, tck)
    return np.asarray(x), np.asarray(y)


def normalize_xy(xy):
    xy = np.asarray(xy, dtype=np.float64)
    p0 = xy[0]
    d = xy[-1] - p0
    length = np.linalg.norm(d)
    if length < 1e-9:
        return np.zeros(len(xy)), np.zeros(len(xy))
    e1 = d / length
    e2 = np.array([-e1[1], e1[0]])
    q = xy - p0
    along = (q @ e1) / length
    lateral = (q @ e2) / length
    return lateral, along


def draw_xy(ax, xy, title, show_ends=True, equal=True):
    x, y = smooth_xy(xy)
    ax.plot(x, y, "-", color="#1f77b4", linewidth=1.8)
    if show_ends:
        ax.scatter(xy[0, 0], xy[0, 1], s=35, color="black", zorder=5)
        ax.scatter(xy[-1, 0], xy[-1, 1], s=35, marker="s", color="black", zorder=5)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("X (m)", fontsize=7)
    ax.set_ylabel("Y (m)", fontsize=7)
    ax.tick_params(labelsize=6)
    if equal:
        ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)


def select_trajectories(traj_dir, spins, limit, min_points, min_duration, spin_limit):
    files = sorted([f for f in traj_dir.glob("*.csv") if f.name != "index.csv"])
    selected = []
    for f in files:
        tid = int(f.stem)
        if tid in spins and np.linalg.norm(spins[tid]) > spin_limit:
            continue
        t, p = read_traj(f)
        bi = first_bounce(t, p)
        if bi is not None:
            if bi < 3:
                continue
            t_use, p_use = t[:bi], p[:bi]
            flag = "bounce"
        else:
            t_use, p_use = t, p
            flag = "no_bounce"
        n = len(t_use)
        dur = float(t_use[-1] - t_use[0]) if n > 1 else 0.0
        if n < min_points or dur < min_duration:
            continue
        dt = np.diff(t_use)
        if len(dt) and np.any(dt > 2.5 * np.median(dt)):
            continue
        selected.append({"id": tid, "t": t_use, "p": p_use, "n": n, "dur": dur, "flag": flag})
        if len(selected) >= limit:
            break
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="trajectories")
    ap.add_argument("--index", default="trajectories/index.csv")
    ap.add_argument("--gt-dir", dest="gt_dir", default="3D_gt")
    ap.add_argument("--out", default="pretrain/output/straightness")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--min-points", type=int, default=30)
    ap.add_argument("--min-duration", type=float, default=0.25)
    ap.add_argument("--spin-limit", type=float, default=628.0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    def rel(p):
        q = Path(p)
        return q if q.is_absolute() else root / q
    traj_dir = rel(args.dir)
    out_dir = rel(args.out)
    xy_dir = out_dir / "trajectories_xy"
    xy_dir.mkdir(parents=True, exist_ok=True)

    spins = load_spin_index(rel(args.index))
    selected = select_trajectories(traj_dir, spins, args.limit, args.min_points, args.min_duration, args.spin_limit)
    print(f"selected {len(selected)} trajectories:")
    for s in selected:
        print(f"  id {s['id']:>3}  n_pre {s['n']:>3}  dur {s['dur']:.3f}  {s['flag']}")

    g3 = sorted(rel(args.gt_dir).glob("*.csv"))[: args.limit]
    g3_data = []
    for f in g3:
        t, p = read_traj(f)
        bi = first_bounce(t, p)
        if bi is not None and bi >= 3:
            t_use, p_use = t[:bi], p[:bi]
        else:
            t_use, p_use = t, p
        g3_data.append({"id": int(f.stem), "p": p_use})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for s in selected:
        fig, ax = plt.subplots(figsize=(6, 7))
        draw_xy(ax, s["p"][:, :2], f"trajectories {s['id']:03d} (n={s['n']}, {s['dur']:.2f}s)")
        fig.tight_layout()
        fig.savefig(xy_dir / f"xy_{s['id']:03d}.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    for ax, s in zip(axes.ravel(), selected):
        draw_xy(ax, s["p"][:, :2], f"trajectories {s['id']:03d}")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectories_all.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    for ax, g in zip(axes.ravel(), g3_data):
        draw_xy(ax, g["p"][:, :2], f"3D_gt {g['id']:03d}")
    fig.tight_layout()
    fig.savefig(out_dir / "3dgt_all.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, args.limit, figsize=(3.0 * args.limit, 7))
    for j, s in enumerate(selected):
        draw_xy(axes[0, j], s["p"][:, :2], f"traj {s['id']:03d}", show_ends=False, equal=True)
    for j, g in enumerate(g3_data):
        draw_xy(axes[1, j], g["p"][:, :2], f"3D_gt {g['id']:03d}", show_ends=False, equal=True)
    axes[0, 0].set_ylabel("trajectories\nY (m)")
    axes[1, 0].set_ylabel("3D_gt\nY (m)")
    fig.tight_layout()
    fig.savefig(out_dir / "compare_3dgt_vs_trajectories.png", dpi=150)
    plt.close(fig)

    dev = []
    for s in selected:
        lat, along = normalize_xy(s["p"][:, :2])
        dev.append((lat, along))
    dev_g = []
    for g in g3_data:
        lat, along = normalize_xy(g["p"][:, :2])
        dev_g.append((lat, along))
    max_abs = max([np.max(np.abs(x[0])) for x in dev + dev_g] + [0.01])
    lim = min(max(0.02, max_abs * 1.1), 0.1)
    fig, axes = plt.subplots(2, args.limit, figsize=(3.0 * args.limit, 6))
    for j, ((lat, along), s) in enumerate(zip(dev, selected)):
        axes[0, j].plot(along, lat, lw=1.4)
        axes[0, j].axhline(0, ls="--", color="gray", lw=0.8)
        axes[0, j].set_ylim(-lim, lim)
        axes[0, j].set_title(f"traj {s['id']:03d}", fontsize=8)
        axes[0, j].grid(alpha=0.25)
    for j, ((lat, along), g) in enumerate(zip(dev_g, g3_data)):
        axes[1, j].plot(along, lat, lw=1.4)
        axes[1, j].axhline(0, ls="--", color="gray", lw=0.8)
        axes[1, j].set_ylim(-lim, lim)
        axes[1, j].set_title(f"3D_gt {g['id']:03d}", fontsize=8)
        axes[1, j].grid(alpha=0.25)
    axes[0, 0].set_ylabel("trajectories\nlateral/chord")
    axes[1, 0].set_ylabel("3D_gt\nlateral/chord")
    for j in range(args.limit):
        axes[1, j].set_xlabel("along chord")
    fig.tight_layout()
    fig.savefig(out_dir / "straightness_deviation.png", dpi=150)
    plt.close(fig)

    print(f"figures saved in {out_dir}")


if __name__ == "__main__":
    main()
