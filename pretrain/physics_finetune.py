import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import SpinCNN
from finetune import rollout, rotate_xy, rotate_back_np, KD, KM
from fit_trajectories_physics import read_traj, first_bounce, select_usable, load_index

L = 64
GRID = np.linspace(0.0, 1.0, L)


def load_v0_map(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[int(r["id"])] = np.array([float(r["v0_x"]), float(r["v0_y"]), float(r["v0_z"])])
    return out


def build_inputs(items, v0_map, index, norm):
    n = len(items)
    X = np.zeros((n, 7, L), dtype=np.float32)
    pos_c = np.zeros((n, L, 3), dtype=np.float32)
    v0_c = np.zeros((n, 3), dtype=np.float32)
    dts = np.zeros(n, dtype=np.float32)
    theta = np.zeros(n, dtype=np.float32)
    v_hat = np.zeros((n, 3), dtype=np.float32)
    gt_w = np.zeros((n, 3), dtype=np.float32)
    ids = np.zeros(n, dtype=np.int64)
    x_mean = np.array(norm["x_mean"], dtype=np.float32).reshape(1, 7, 1)
    x_std = np.array(norm["x_std"], dtype=np.float32).reshape(1, 7, 1)
    for i, it in enumerate(items):
        t, p = it["t"], it["p"]
        T = t[-1] - t[0]
        v = np.gradient(p, t, axis=0)
        th = float(np.arctan2(v[0, 0], v[0, 1]))
        p_c = rotate_xy(p - p[0], th)
        v_c = rotate_xy(v, th)
        tau = (t - t[0]) / max(T, 1e-12)
        p_r = np.column_stack([np.interp(GRID, tau, p_c[:, j]) for j in range(3)])
        v_r = np.column_stack([np.interp(GRID, tau, v_c[:, j]) for j in range(3)])
        X[i, :3] = p_r.T
        X[i, 3:6] = v_r.T
        X[i, 6] = GRID
        pos_c[i] = p_r
        v0_c[i] = rotate_xy(v0_map[it["id"]][None, :], th)[0]
        dts[i] = T / (L - 1)
        theta[i] = th
        vm = v.mean(axis=0)
        vm = vm / max(np.linalg.norm(vm), 1e-12)
        v_hat[i] = vm
        gt_w[i] = index.get(it["id"], np.array([np.nan, np.nan, np.nan]))
        ids[i] = it["id"]
    X = (X - x_mean) / x_std
    return {
        "X": torch.from_numpy(X.astype(np.float32)),
        "pos": torch.from_numpy(pos_c),
        "v0": torch.from_numpy(v0_c),
        "dts": torch.from_numpy(dts),
        "theta": theta,
        "v_hat": v_hat,
        "gt": gt_w,
        "ids": ids,
    }


def pred_world(model, X, theta, y_mean, y_std):
    model.eval()
    with torch.no_grad():
        pred_n = model(X).numpy()
    w_c = pred_n * y_std + y_mean
    return rotate_back_np(w_c, theta)


def rollout_rmse(model, X, pos, v0, dts, y_mean, y_std):
    model.eval()
    with torch.no_grad():
        pred_n = model(X)
        w = pred_n * y_std + y_mean
        rp = rollout(v0, w, dts, L, KD, KM)
        rmse = torch.sqrt(((rp - pos) ** 2).sum(dim=2)).mean().item() * 100.0
    return rmse


def angle_deg(pred, true):
    pn = np.linalg.norm(pred, axis=1)
    tn = np.linalg.norm(true, axis=1)
    c = np.clip(np.sum(pred * true, axis=1) / np.maximum(pn * tn, 1e-12), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def perp(v, vhat):
    dot = np.sum(v * vhat, axis=1, keepdims=True)
    return v - dot * vhat


def metrics(w_world, gt_w, v_hat, mask):
    w = w_world[mask]
    g = gt_w[mask]
    vh = v_hat[mask]
    full_diff = w - g
    wp = perp(w, vh)
    gp = perp(g, vh)
    full_ang = angle_deg(w, g)
    perp_ang = angle_deg(wp, gp)
    return {
        "n": int(mask.sum()),
        "full_mae": float(np.mean(np.abs(full_diff))),
        "full_angle": float(np.mean(full_ang)),
        "perp_mae": float(np.mean(np.abs(wp - gp))),
        "perp_angle": float(np.mean(perp_ang)),
        "perp_angle_median": float(np.median(perp_ang)),
        "full_norm_mae": float(np.mean(np.abs(np.linalg.norm(w, axis=1) - np.linalg.norm(g, axis=1)))),
    }


def train_head(model, Xtr, pos_tr, v0_tr, dts_tr, Xva, pos_va, v0_va, dts_va, epochs, patience, lr, l2, batch_size, seed, y_mean_np, y_std_np):
    torch.manual_seed(seed)
    y_mean_t = torch.tensor(y_mean_np, dtype=torch.float32).view(1, 3)
    y_std_t = torch.tensor(y_std_np, dtype=torch.float32).view(1, 3)
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    head_params = list(model.head.parameters())
    anchor = {name: p.detach().clone() for name, p in model.head.named_parameters()}
    optimizer = torch.optim.AdamW(head_params, lr=lr, weight_decay=1e-3)
    ds = TensorDataset(Xtr, pos_tr, v0_tr, dts_tr)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    best_rmse = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = -1
    wait = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, pb, vb, db in loader:
            optimizer.zero_grad(set_to_none=True)
            pred_n = model(xb)
            w = pred_n * y_std_t + y_mean_t
            rp = rollout(vb, w, db, L, KD, KM)
            loss = F.huber_loss(rp, pb, delta=0.02)
            reg = 0.0
            for name, p in model.head.named_parameters():
                reg = reg + ((p - anchor[name]) ** 2).mean()
            loss = loss + l2 * reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 1.0)
            optimizer.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)
        val_rmse = rollout_rmse(model, Xva, pos_va, v0_va, dts_va, y_mean_np, y_std_np)
        if val_rmse < best_rmse - 1e-5:
            best_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"    epoch {epoch:3d} | loss {total/max(count,1):.5f} | val rollout RMSE {val_rmse:.2f} cm")
        if wait >= patience:
            print(f"    early stop at {epoch}, best {best_epoch}")
            break
    model.load_state_dict(best_state)
    return best_rmse, best_epoch


def run_for_checkpoint(ckpt_path, data, index, out_dir, args):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base_state = ckpt["state_dict"]
    norm = ckpt["norm"]
    y_mean_np = np.array(norm["y_mean"], dtype=np.float32)
    y_std_np = np.array(norm["y_std"], dtype=np.float32)

    X = data["X"]
    pos = data["pos"]
    v0 = data["v0"]
    dts = data["dts"]
    theta = data["theta"]
    v_hat = data["v_hat"]
    gt = data["gt"]
    ids = data["ids"]
    n = len(ids)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    folds = np.empty(n, dtype=int)
    for rank, idx in enumerate(order):
        folds[idx] = rank % args.folds

    base_model = SpinModel(base_state)
    base_w = pred_world(base_model, X, theta, y_mean_np, y_std_np)
    base_rmse_all = rollout_rmse(base_model, X, pos, v0, dts, y_mean_np, y_std_np)

    results = []
    oof_pred = np.full((n, 3), np.nan)
    for k in range(args.folds):
        test_idx = np.where(folds == k)[0]
        val_idx = np.where(folds == (k + 1) % args.folds)[0]
        train_idx = np.where((folds != k) & (folds != (k + 1) % args.folds))[0]
        print(f"  fold {k}: train {len(train_idx)} val {len(val_idx)} test {len(test_idx)}")
        model = SpinModel(base_state)
        best_rmse, best_epoch = train_head(
            model,
            X[train_idx], pos[train_idx], v0[train_idx], dts[train_idx],
            X[val_idx], pos[val_idx], v0[val_idx], dts[val_idx],
            args.epochs, args.patience, args.lr, args.l2, args.batch_size, args.seed + k,
            y_mean_np, y_std_np,
        )
        w_test = pred_world(model, X[test_idx], theta[test_idx], y_mean_np, y_std_np)
        oof_pred[test_idx] = w_test
        test_rmse = rollout_rmse(model, X[test_idx], pos[test_idx], v0[test_idx], dts[test_idx], y_mean_np, y_std_np)
        base_test_rmse = rollout_rmse(base_model, X[test_idx], pos[test_idx], v0[test_idx], dts[test_idx], y_mean_np, y_std_np)
        mask = np.isfinite(gt[test_idx]).all(axis=1) & (np.linalg.norm(gt[test_idx], axis=1) <= 628.0)
        ft_m = metrics(w_test, gt[test_idx], v_hat[test_idx], mask)
        base_m = metrics(base_w[test_idx], gt[test_idx], v_hat[test_idx], mask)
        results.append({
            "fold": k,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "best_epoch": int(best_epoch),
            "base_rollout_rmse": float(base_test_rmse),
            "ft_rollout_rmse": float(test_rmse),
            "base": base_m,
            "finetuned": ft_m,
        })
        print(f"    rollout RMSE {base_test_rmse:.2f} -> {test_rmse:.2f} cm | "
              f"perp angle {base_m['perp_angle']:.1f} -> {ft_m['perp_angle']:.1f} | "
              f"full angle {base_m['full_angle']:.1f} -> {ft_m['full_angle']:.1f}")

    def mean_std(key, sub):
        v = np.array([r[key][sub] for r in results])
        return float(v.mean()), float(v.std())

    summary = {
        "checkpoint": str(ckpt_path),
        "n_usable": int(n),
        "base_all_rollout_rmse": float(base_rmse_all),
        "cv": {
            "base_rollout_mean": float(np.mean([r["base_rollout_rmse"] for r in results])),
            "base_rollout_std": float(np.std([r["base_rollout_rmse"] for r in results])),
            "ft_rollout_mean": float(np.mean([r["ft_rollout_rmse"] for r in results])),
            "ft_rollout_std": float(np.std([r["ft_rollout_rmse"] for r in results])),
            "base_perp_angle_mean": float(np.mean([r["base"]["perp_angle"] for r in results])),
            "base_perp_angle_std": float(np.std([r["base"]["perp_angle"] for r in results])),
            "ft_perp_angle_mean": float(np.mean([r["finetuned"]["perp_angle"] for r in results])),
            "ft_perp_angle_std": float(np.std([r["finetuned"]["perp_angle"] for r in results])),
            "base_perp_mae_mean": float(np.mean([r["base"]["perp_mae"] for r in results])),
            "ft_perp_mae_mean": float(np.mean([r["finetuned"]["perp_mae"] for r in results])),
            "base_full_angle_mean": float(np.mean([r["base"]["full_angle"] for r in results])),
            "ft_full_angle_mean": float(np.mean([r["finetuned"]["full_angle"] for r in results])),
        },
        "fold_results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cv_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    with open(out_dir / "cv_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "base_rollout", "ft_rollout",
                    "base_full_angle", "ft_full_angle", "base_perp_angle", "ft_perp_angle",
                    "base_perp_mae", "ft_perp_mae"])
        for r in results:
            w.writerow([r["fold"], r["base_rollout_rmse"], r["ft_rollout_rmse"],
                        r["base"]["full_angle"], r["finetuned"]["full_angle"],
                        r["base"]["perp_angle"], r["finetuned"]["perp_angle"],
                        r["base"]["perp_mae"], r["finetuned"]["perp_mae"]])
    with open(out_dir / "oof_predictions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "pred_x", "pred_y", "pred_z", "gt_x", "gt_y", "gt_z",
                    "pred_perp_x", "pred_perp_y", "pred_perp_z",
                    "gt_perp_x", "gt_perp_y", "gt_perp_z", "perp_angle"])
        for i in range(n):
            vh = v_hat[i]
            pp = oof_pred[i] - np.dot(oof_pred[i], vh) * vh
            gp = gt[i] - np.dot(gt[i], vh) * vh
            a = angle_deg(pp[None, :], gp[None, :])[0]
            w.writerow([int(ids[i]), oof_pred[i, 0], oof_pred[i, 1], oof_pred[i, 2],
                        gt[i, 0], gt[i, 1], gt[i, 2],
                        pp[0], pp[1], pp[2], gp[0], gp[1], gp[2], a])
    return summary


def SpinModel(state_dict):
    model = SpinCNN(in_ch=7, out=3)
    model.load_state_dict(state_dict)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="trajectories")
    ap.add_argument("--index", default="trajectories/index.csv")
    ap.add_argument("--v0-csv", default="pretrain/output/fit_trajectories_physics/fit_results.csv")
    ap.add_argument("--ckpts", default="pretrain/output/finetune/finetuned_model.pt,pretrain/output/best_model.pt")
    ap.add_argument("--out", default="pretrain/output/physics_finetune")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-points", type=int, default=30)
    ap.add_argument("--min-duration", type=float, default=0.25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1.0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    def rel(p):
        q = Path(p)
        return q if q.is_absolute() else root / q
    out_dir = rel(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(4)
    index = load_index(rel(args.index))
    v0_map = load_v0_map(rel(args.v0_csv))
    items = select_usable(rel(args.dir), args.min_points, args.min_duration)
    items = [it for it in items if it["id"] in v0_map]
    print(f"usable trajectories: {len(items)}")

    ckpts = [Path(x) for x in args.ckpts.split(",")]
    summaries = {}
    for ckpt in ckpts:
        ckpt_path = rel(ckpt)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        norm = ckpt["norm"]
        data = build_inputs(items, v0_map, index, norm)
        name = ckpt_path.parent.name + "_" + ckpt_path.stem
        print(f"\n=== {name} ===")
        sub = out_dir / name
        summaries[name] = run_for_checkpoint(ckpt_path, data, index, sub, args)

    with open(out_dir / "physics_finetune_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, ensure_ascii=False, indent=2)

    print("\noverall summary")
    for name, s in summaries.items():
        c = s["cv"]
        print(f"  {name}: rollout {c['base_rollout_mean']:.2f} -> {c['ft_rollout_mean']:.2f} cm | "
              f"perp angle {c['base_perp_angle_mean']:.1f} -> {c['ft_perp_angle_mean']:.1f} | "
              f"full angle {c['base_full_angle_mean']:.1f} -> {c['ft_full_angle_mean']:.1f}")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
