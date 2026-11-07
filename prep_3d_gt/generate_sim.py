import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from scipy.stats.qmc import LatinHypercube

G = np.array([0.0, 0.0, -9.81])
R = 0.02
M = 2.7e-3
TABLE_HALF_LENGTH = 1.37
TABLE_HALF_WIDTH = 0.7625
NET_HEIGHT = 0.1525
NET_HALF_WIDTH = 0.915
CLEARANCE = NET_HEIGHT + R
DT = 0.0056
KD_DEFAULT = 3.9827679117703966e-4
KM_DEFAULT = 4.86e-6


def load_meta(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def accel(v, w, kd, km):
    speed = np.linalg.norm(v)
    return G - (kd / M) * speed * v + (km / M) * np.cross(w, v)


def simulate(r0, v0, w, dt, kd, km, max_t=1.5):
    r = np.array(r0, dtype=float)
    v = np.array(v0, dtype=float)
    speed0 = np.linalg.norm(v)
    if not np.isfinite(r).all() or not np.isfinite(v).all() or not np.isfinite(w).all():
        return None
    if speed0 < 2.0 or speed0 > 15.0:
        return None
    if np.linalg.norm(w) > 1000.0:
        return None

    n_max = int(max_t / dt) + 1
    p = np.empty((n_max, 3))
    p[0] = r
    k = None
    for i in range(1, n_max):
        k1v = accel(v, w, kd, km)
        k1r = v
        k2v = accel(v + 0.5 * dt * k1v, w, kd, km)
        k2r = v + 0.5 * dt * k1v
        k3v = accel(v + 0.5 * dt * k2v, w, kd, km)
        k3r = v + 0.5 * dt * k2v
        k4v = accel(v + dt * k3v, w, kd, km)
        k4r = v + dt * k3v
        v = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        r = r + dt / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
        p[i] = r
        if r[2] <= R:
            k = i
            break
    if k is None or k < 6:
        return None

    y = p[:, 1]
    sign = 1.0 if v0[1] > 0 else -1.0
    net_i = None
    for i in range(1, k + 1):
        if y[i - 1] * y[i] <= 0:
            denom = y[i - 1] - y[i]
            alpha = 0.0 if denom == 0 else y[i - 1] / denom
            x_net = p[i - 1, 0] + alpha * (p[i, 0] - p[i - 1, 0])
            z_net = p[i - 1, 2] + alpha * (p[i, 2] - p[i - 1, 2])
            net_i = i
            break
    if net_i is None or net_i >= k:
        return None
    if z_net <= CLEARANCE or abs(x_net) > NET_HALF_WIDTH:
        return None

    z0 = p[k - 1, 2]
    z1 = p[k, 2]
    denom = z0 - z1
    alpha = 0.0 if denom == 0 else (z0 - R) / denom
    landing = p[k - 1] + alpha * (p[k] - p[k - 1])
    if sign * landing[1] <= 0:
        return None
    if abs(landing[1]) > TABLE_HALF_LENGTH or abs(landing[0]) > TABLE_HALF_WIDTH:
        return None

    t_land = (k - 1 + alpha) * dt
    if t_land < 0.15 or t_land > 1.0:
        return None

    t_out = np.arange(k) * dt
    p_out = p[:k]
    if len(p_out) < 6:
        return None
    return t_out, p_out, landing, t_land, float(z_net), float(x_net)


def perturb_seed(seed, rng):
    r0 = np.array([float(seed["r0_x"]), float(seed["r0_y"]), float(seed["r0_z"])])
    v0 = np.array([float(seed["v0_x"]), float(seed["v0_y"]), float(seed["v0_z"])])
    w = np.array([float(seed["omega_x"]), float(seed["omega_y"]), float(seed["omega_z"])])
    r0 = r0 + rng.normal(0.0, [0.03, 0.04, 0.02], size=3)
    r0[2] = max(r0[2], R + 0.02)
    sf = rng.uniform(0.92, 1.08)
    v0 = v0 * sf + rng.normal(0.0, 0.04 * np.linalg.norm(v0), size=3)
    wf = rng.uniform(0.90, 1.10)
    w = w * wf + rng.normal(0.0, 0.10 * np.linalg.norm(w), size=3)
    wn = np.linalg.norm(w)
    if wn > 1000.0:
        w = w / wn * 1000.0
    return r0, v0, w


def sample_lhs(n, rng):
    sampler = LatinHypercube(d=11, seed=rng)
    u = sampler.random(n)
    s = np.where(u[:, 0] < 0.575, 1.0, -1.0)
    speed = 2.0 + u[:, 1] * 13.0
    elev = np.radians(-5.0 + u[:, 2] * 45.0)
    azim = np.radians(-25.0 + u[:, 3] * 50.0)
    x0 = -0.7 + u[:, 4] * 1.4
    y0_abs = 1.0 + u[:, 5] * 0.9
    z0 = 0.05 + u[:, 6] * 0.45
    spin_mag = 50.0 + u[:, 7] * 850.0
    alpha = np.radians(u[:, 8] * 80.0)
    side_sign = np.where(u[:, 9] < 0.5, 1.0, -1.0)
    spin_sign = np.where(u[:, 10] < 0.5, 1.0, -1.0)

    vx = speed * np.cos(elev) * np.sin(azim)
    vy = speed * np.cos(elev) * np.cos(azim) * s
    vz = speed * np.sin(elev)
    v0_all = np.column_stack([vx, vy, vz])
    r0_all = np.column_stack([x0, -s * y0_abs, z0])

    vhat = v0_all / np.linalg.norm(v0_all, axis=1, keepdims=True)
    zhat = np.array([0.0, 0.0, 1.0])
    nvec = np.cross(vhat, zhat)
    nn = np.linalg.norm(nvec, axis=1, keepdims=True)
    fallback = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    nvec = np.where(nn > 1e-8, nvec / np.maximum(nn, 1e-12), fallback)
    n2 = np.cross(vhat, nvec)
    n2 = n2 / np.linalg.norm(n2, axis=1, keepdims=True)

    ax = np.cos(alpha)[:, None] * nvec * spin_sign[:, None] + np.sin(alpha)[:, None] * n2 * side_sign[:, None]
    ax = ax / np.linalg.norm(ax, axis=1, keepdims=True)
    w_all = ax * spin_mag[:, None]
    return r0_all, v0_all, w_all


def split_seed_ids(seed_ids, spins):
    order = np.argsort(spins)
    split = {}
    for rank, idx in enumerate(order):
        sid = seed_ids[idx]
        if rank % 7 == 5:
            split[sid] = "val"
        elif rank % 7 == 6:
            split[sid] = "test"
        else:
            split[sid] = "train"
    return split


def write_traj(path, t, p):
    arr = np.column_stack([t, p])
    np.savetxt(path, arr, delimiter=",", header="Timestamp,X,Y,Z", comments="", fmt="%.6f")


def run(args):
    root = Path(__file__).resolve().parent.parent
    meta_path = Path(args.meta)
    if not meta_path.is_absolute():
        meta_path = root / meta_path
    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    for old in traj_dir.glob("*.csv"):
        old.unlink()

    kd = args.kd
    if kd is None:
        calib_path = root / "prep_3d_gt" / "output" / "calibration.json"
        if calib_path.exists():
            with open(calib_path, encoding="utf-8") as fh:
                calib = json.load(fh)
            it = calib.get("iterations", [])
            kd = float(it[-1]["kd_after"]) if it else float(calib.get("kd_initial", KD_DEFAULT))
        else:
            kd = KD_DEFAULT
    km = args.km
    dt = args.dt
    rng = np.random.default_rng(args.seed)

    rows = load_meta(meta_path)
    usable = [r for r in rows if int(r["unusable"]) == 0]
    train_seeds = [r for r in usable if r.get("split") == "train"]
    if not args.include_near_bound:
        train_seeds = [r for r in train_seeds if "omega_near_bound" not in r.get("flags", "")]
    if args.limit_seeds:
        train_seeds = train_seeds[: args.limit_seeds]
    print(f"seed trajectories: {len(train_seeds)}  kd={kd:.6g} km={km:.6g}")

    seed_splits = split_seed_ids(
        [int(r["id"]) for r in train_seeds],
        [float(r["omega_norm"]) for r in train_seeds],
    )

    sim_rows = []
    counters = {"seed_attempts": 0, "seed_accepted": 0, "lhs_attempts": 0, "lhs_accepted": 0}
    next_id = 0

    for seed in train_seeds:
        sid = int(seed["id"])
        target = args.variants_per_seed
        kept = 0
        attempts = 0
        max_attempts = target * 15
        while kept < target and attempts < max_attempts:
            attempts += 1
            r0, v0, w = perturb_seed(seed, rng)
            sim = simulate(r0, v0, w, dt, kd, km)
            if sim is None:
                continue
            t_out, p_out, landing, t_land, z_net, x_net = sim
            split = seed_splits[sid]
            write_traj(traj_dir / f"sim_{next_id:05d}.csv", t_out, p_out)
            sim_rows.append({
                "id": next_id,
                "source": "seed",
                "seed_id": sid,
                "split": split,
                "n_points": len(t_out),
                "duration": t_land,
                "r0_x": r0[0], "r0_y": r0[1], "r0_z": r0[2],
                "v0_x": v0[0], "v0_y": v0[1], "v0_z": v0[2],
                "omega_x": w[0], "omega_y": w[1], "omega_z": w[2],
                "omega_norm": float(np.linalg.norm(w)),
                "landing_x": landing[0], "landing_y": landing[1], "landing_z": landing[2],
                "net_clearance": z_net,
                "net_x": x_net,
                "path_length": float(np.sum(np.linalg.norm(np.diff(p_out, axis=0), axis=1))),
                "kd": kd, "km": km,
            })
            next_id += 1
            kept += 1
        counters["seed_attempts"] += attempts
        counters["seed_accepted"] += kept
        if kept < target:
            print(f"  seed {sid}: only {kept}/{target} accepted ({attempts} attempts)")

    lhs_target = args.lhs_target
    while counters["lhs_accepted"] < lhs_target and counters["lhs_attempts"] < lhs_target * 25:
        batch = min(500, max(100, lhs_target - counters["lhs_accepted"]))
        r0_all, v0_all, w_all = sample_lhs(batch, rng)
        counters["lhs_attempts"] += batch
        for r0, v0, w in zip(r0_all, v0_all, w_all):
            if counters["lhs_accepted"] >= lhs_target:
                break
            sim = simulate(r0, v0, w, dt, kd, km)
            if sim is None:
                continue
            t_out, p_out, landing, t_land, z_net, x_net = sim
            if rng.random() < 0.70:
                split = "train"
            elif rng.random() < 0.5:
                split = "val"
            else:
                split = "test"
            write_traj(traj_dir / f"sim_{next_id:05d}.csv", t_out, p_out)
            sim_rows.append({
                "id": next_id,
                "source": "lhs",
                "seed_id": -1,
                "split": split,
                "n_points": len(t_out),
                "duration": t_land,
                "r0_x": r0[0], "r0_y": r0[1], "r0_z": r0[2],
                "v0_x": v0[0], "v0_y": v0[1], "v0_z": v0[2],
                "omega_x": w[0], "omega_y": w[1], "omega_z": w[2],
                "omega_norm": float(np.linalg.norm(w)),
                "landing_x": landing[0], "landing_y": landing[1], "landing_z": landing[2],
                "net_clearance": z_net,
                "net_x": x_net,
                "path_length": float(np.sum(np.linalg.norm(np.diff(p_out, axis=0), axis=1))),
                "kd": kd, "km": km,
            })
            next_id += 1
            counters["lhs_accepted"] += 1

    meta_out = out_dir / "sim_meta.csv"
    fields = [
        "id", "source", "seed_id", "split", "n_points", "duration",
        "r0_x", "r0_y", "r0_z", "v0_x", "v0_y", "v0_z",
        "omega_x", "omega_y", "omega_z", "omega_norm",
        "landing_x", "landing_y", "landing_z", "net_clearance", "net_x",
        "path_length", "kd", "km",
    ]
    with open(meta_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sim_rows:
            w.writerow({k: r[k] for k in fields})

    split_out = out_dir / "sim_splits.csv"
    with open(split_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "source", "seed_id", "split"])
        for r in sim_rows:
            w.writerow([r["id"], r["source"], r["seed_id"], r["split"]])

    real_train = [r for r in usable if r.get("split") == "train"]
    features = {
        "omega_norm": (
            np.array([float(r["omega_norm"]) for r in real_train]),
            np.array([r["omega_norm"] for r in sim_rows]),
        ),
        "speed": (
            np.array([np.linalg.norm([float(r["v0_x"]), float(r["v0_y"]), float(r["v0_z"])]) for r in real_train]),
            np.array([np.linalg.norm([r["v0_x"], r["v0_y"], r["v0_z"]]) for r in sim_rows]),
        ),
        "duration": (
            np.array([float(r["duration"]) for r in real_train]),
            np.array([r["duration"] for r in sim_rows]),
        ),
        "r0_x": (
            np.array([float(r["r0_x"]) for r in real_train]),
            np.array([r["r0_x"] for r in sim_rows]),
        ),
        "r0_y": (
            np.array([float(r["r0_y"]) for r in real_train]),
            np.array([r["r0_y"] for r in sim_rows]),
        ),
        "r0_z": (
            np.array([float(r["r0_z"]) for r in real_train]),
            np.array([r["r0_z"] for r in sim_rows]),
        ),
        "landing_x": (
            np.array([float(r["landing_x"]) for r in real_train]),
            np.array([r["landing_x"] for r in sim_rows]),
        ),
        "landing_y": (
            np.array([float(r["landing_y"]) for r in real_train]),
            np.array([r["landing_y"] for r in sim_rows]),
        ),
    }

    report = {
        "n_seed_traj": int(sum(1 for r in sim_rows if r["source"] == "seed")),
        "n_lhs_traj": int(sum(1 for r in sim_rows if r["source"] == "lhs")),
        "n_total": len(sim_rows),
        "kd": kd,
        "km": km,
        "seed_acceptance": counters["seed_accepted"] / max(counters["seed_attempts"], 1),
        "lhs_acceptance": counters["lhs_accepted"] / max(counters["lhs_attempts"], 1),
        "seed_splits": {k: int(sum(1 for s in seed_splits.values() if s == k)) for k in ("train", "val", "test")},
        "sim_split_counts": {k: int(sum(1 for r in sim_rows if r["split"] == k)) for k in ("train", "val", "test")},
        "features": {},
    }
    for name, (real, sim) in features.items():
        ks = ks_2samp(real, sim)
        report["features"][name] = {
            "real_median": float(np.median(real)),
            "real_p5": float(np.percentile(real, 5)),
            "real_p95": float(np.percentile(real, 95)),
            "sim_median": float(np.median(sim)),
            "sim_p5": float(np.percentile(sim, 5)),
            "sim_p95": float(np.percentile(sim, 95)),
            "ks_stat": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
        }

    with open(out_dir / "report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        for ax, (name, (real, sim)) in zip(axes.ravel(), features.items()):
            bins = np.histogram_bin_edges(np.concatenate([real, sim]), bins=25)
            ax.hist(real, bins=bins, density=True, alpha=0.5, label="3D_gt train")
            ax.hist(sim, bins=bins, density=True, alpha=0.5, label="sim")
            ax.set_title(f"{name}  KS={report['features'][name]['ks_stat']:.2f}")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "coverage.png", dpi=150)
        plt.close(fig)

    print("\nsummary")
    print(f"  seed trajectories: {report['n_seed_traj']}  lhs: {report['n_lhs_traj']}  total: {report['n_total']}")
    print(f"  seed acceptance: {report['seed_acceptance']*100:.1f}%  lhs acceptance: {report['lhs_acceptance']*100:.1f}%")
    print(f"  seed splits: {report['seed_splits']}")
    print(f"  sim split counts: {report['sim_split_counts']}")
    for name in ("omega_norm", "speed", "duration", "landing_y"):
        f = report["features"][name]
        print(f"  {name}: real med {f['real_median']:.2f} vs sim {f['sim_median']:.2f}  KS {f['ks_stat']:.2f}")
    print(f"  output: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="prep_3d_gt/output/3D_gt_meta.csv")
    ap.add_argument("--output", default="prep_3d_gt/output/sim_pretrain")
    ap.add_argument("--dt", type=float, default=DT)
    ap.add_argument("--kd", type=float, default=None)
    ap.add_argument("--km", type=float, default=KM_DEFAULT)
    ap.add_argument("--variants-per-seed", type=int, default=20)
    ap.add_argument("--lhs-target", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-seeds", type=int, default=0)
    ap.add_argument("--include-near-bound", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
