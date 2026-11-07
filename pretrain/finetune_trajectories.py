import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import SpinCNN
from finetune import build_split, rollout, KD, KM, rotate_xy, rotate_back_np

L = 64
GRID = np.linspace(0.0, 1.0, L)
SPIN_LIMIT = 628.0


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
    spins = {}
    with open(path, newline="", encoding="utf-8") as fh:
        r = csv.reader(fh)
        next(r)
        for row in r:
            if len(row) < 4:
                continue
            spins[int(row[0])] = np.array([float(row[1]), float(row[2]), float(row[3])])
    return spins


def select_usable(traj_dir, spins, min_points, min_duration, spin_limit):
    files = sorted([f for f in traj_dir.glob("*.csv") if f.name != "index.csv"])
    out = []
    for f in files:
        tid = int(f.stem)
        if tid not in spins:
            continue
        gt = spins[tid]
        if np.linalg.norm(gt) > spin_limit:
            continue
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
        out.append({"id": tid, "t": t_use, "p": p_use, "gt": gt})
    return out


def build_arrays(items, norm):
    n = len(items)
    X = np.zeros((n, 7, L), dtype=np.float32)
    gt_c = np.zeros((n, 3), dtype=np.float32)
    gt_w = np.zeros((n, 3), dtype=np.float32)
    theta = np.zeros(n, dtype=np.float32)
    ids = np.zeros(n, dtype=np.int64)
    x_mean = np.array(norm["x_mean"], dtype=np.float32).reshape(1, 7, 1)
    x_std = np.array(norm["x_std"], dtype=np.float32).reshape(1, 7, 1)
    y_mean = np.array(norm["y_mean"], dtype=np.float32)
    y_std = np.array(norm["y_std"], dtype=np.float32)
    for i, it in enumerate(items):
        t = it["t"]
        p = it["p"]
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
        gt_w[i] = it["gt"]
        gt_c[i] = rotate_xy(it["gt"][None, :], th)[0]
        theta[i] = th
        ids[i] = it["id"]
    X = (X - x_mean) / x_std
    gt_n = (gt_c - y_mean) / y_std
    return X, gt_n, gt_c, gt_w, theta, ids


def angle_error(pred, true):
    pn = np.linalg.norm(pred, axis=1)
    tn = np.linalg.norm(true, axis=1)
    cosv = np.clip(np.sum(pred * true, axis=1) / np.maximum(pn * tn, 1e-12), -1.0, 1.0)
    return np.degrees(np.arccos(cosv))


def spin_metrics(pred_w, gt_w):
    diff = pred_w - gt_w
    ang = angle_error(pred_w, gt_w)
    return {
        "mae_x": float(np.mean(np.abs(diff[:, 0]))),
        "mae_y": float(np.mean(np.abs(diff[:, 1]))),
        "mae_z": float(np.mean(np.abs(diff[:, 2]))),
        "mae_overall": float(np.mean(np.abs(diff))),
        "rmse_overall": float(np.sqrt(np.mean(diff ** 2))),
        "mag_mae": float(np.mean(np.abs(np.linalg.norm(pred_w, axis=1) - np.linalg.norm(gt_w, axis=1)))),
        "angle_mean": float(np.mean(ang)),
        "angle_median": float(np.median(ang)),
        "angle_p95": float(np.percentile(ang, 95)),
    }


def predict_world(model, X, theta, y_mean, y_std):
    model.eval()
    with torch.no_grad():
        pred_n = model(torch.from_numpy(X)).numpy()
    pred_c = pred_n * y_std + y_mean
    return rotate_back_np(pred_c, theta)


def make_folds(n, spins, k, seed):
    rng = np.random.default_rng(seed)
    order = np.argsort(spins)
    folds = np.empty(n, dtype=int)
    for rank, idx in enumerate(order):
        folds[idx] = rank % k
    for f in range(k):
        idxs = np.where(folds == f)[0]
        rng.shuffle(idxs)
    return folds


def train_stage(model, Xtr, Ytr, Xva, gt_w_va, theta_va, y_mean, y_std, epochs, patience, lr_groups, anchor, l2, batch_size, seed, tag):
    torch.manual_seed(seed)
    groups = []
    params = []
    for ps, lr in lr_groups:
        ps = [p for p in ps if p.requires_grad]
        if not ps:
            continue
        groups.append({"params": ps, "lr": lr})
        params.extend(ps)
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-3)
    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    best_angle = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = -1
    wait = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.huber_loss(pred, yb, delta=1.0)
            if l2 > 0 and anchor:
                reg = 0.0
                for name, p in model.named_parameters():
                    if p.requires_grad and name in anchor:
                        reg = reg + ((p - anchor[name]) ** 2).mean()
                loss = loss + l2 * reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)
        pred_w = predict_world(model, Xva, theta_va, y_mean, y_std)
        vm = spin_metrics(pred_w, gt_w_va)
        ang = vm["angle_mean"]
        if ang < best_angle - 1e-4:
            best_angle = ang
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  [{tag}] epoch {epoch:3d} | loss {total/max(count,1):.4f} | val angle {ang:.2f} | val mae {vm['mae_overall']:.1f}")
        if wait >= patience:
            print(f"  [{tag}] early stop at {epoch}, best {best_epoch}")
            break
    model.load_state_dict(best_state)
    return best_angle, best_epoch


def capture_anchor(model):
    return {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}


def make_model(state_dict):
    model = SpinCNN(in_ch=7, out=3)
    model.load_state_dict(state_dict)
    return model


def eval_3dgt(model, data, y_mean, y_std):
    model.eval()
    with torch.no_grad():
        pred_n = model(data["X"]).numpy()
    w = pred_n * y_std + y_mean
    w_t = torch.tensor(w, dtype=torch.float32)
    rp = rollout(data["v0_fit"], w_t, data["dts"], L, KD, KM)
    pos = data["pos"]
    return float(torch.sqrt(((rp - pos) ** 2).sum(dim=2)).mean() * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="trajectories")
    ap.add_argument("--index", default="trajectories/index.csv")
    ap.add_argument("--checkpoint", default="pretrain/output/finetune/finetuned_model.pt")
    ap.add_argument("--meta", default="prep_3d_gt/output/3D_gt_meta.csv")
    ap.add_argument("--clean-dir", default="prep_3d_gt/output/3D_gt_clean")
    ap.add_argument("--out", default="pretrain/output/finetune_trajectories")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-points", type=int, default=30)
    ap.add_argument("--min-duration", type=float, default=0.25)
    ap.add_argument("--spin-limit", type=float, default=SPIN_LIMIT)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--stage1-epochs", type=int, default=60)
    ap.add_argument("--stage2-epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--stage1-lr", type=float, default=1e-3)
    ap.add_argument("--stage2-lr", type=float, default=1e-5)
    ap.add_argument("--stage2-lr-head", type=float, default=1e-4)
    ap.add_argument("--unfreeze", choices=["last", "all"], default="last")
    ap.add_argument("--l2", type=float, default=1e-3)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    def rel(p):
        q = Path(p)
        return q if q.is_absolute() else root / q
    out_dir = rel(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(4)
    ckpt = torch.load(rel(args.checkpoint), map_location="cpu", weights_only=False)
    norm = ckpt["norm"]
    base_state = ckpt["state_dict"]
    y_mean = np.array(norm["y_mean"], dtype=np.float32)
    y_std = np.array(norm["y_std"], dtype=np.float32)

    spins = load_index(rel(args.index))
    items = select_usable(rel(args.dir), spins, args.min_points, args.min_duration, args.spin_limit)
    print(f"usable trajectories: {len(items)}")
    if len(items) < 20:
        raise SystemExit("too few usable trajectories")

    X, gt_n, gt_c, gt_w, theta, ids = build_arrays(items, norm)
    n = len(items)
    spins_norm = np.linalg.norm(gt_c, axis=1)
    folds = make_folds(n, spins_norm, args.folds, args.seed)

    base_model = make_model(base_state)
    base_pred_w = predict_world(base_model, X, theta, y_mean, y_std)
    base_all = spin_metrics(base_pred_w, gt_w)

    meta_rows = [r for r in csv.DictReader(open(rel(args.meta), encoding="utf-8"))
                 if int(r["unusable"]) == 0 and r.get("split") == "test"]
    gt_data = build_split(meta_rows, rel(args.clean_dir), L, norm)
    base_3dgt = eval_3dgt(base_model, gt_data, y_mean, y_std)
    print(f"base model: trajectories MAE {base_all['mae_overall']:.1f} | angle {base_all['angle_mean']:.1f} | 3D_gt test pos {base_3dgt:.2f} cm")

    fold_results = []
    oof_pred = np.zeros_like(gt_w)
    for k in range(args.folds):
        test_idx = np.where(folds == k)[0]
        val_idx = np.where(folds == (k + 1) % args.folds)[0]
        train_idx = np.where((folds != k) & (folds != (k + 1) % args.folds))[0]
        print(f"fold {k}: train {len(train_idx)} val {len(val_idx)} test {len(test_idx)}")

        model = make_model(base_state)

        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
        anchor1 = capture_anchor(model)
        train_stage(model, X[train_idx], gt_n[train_idx], X[val_idx], gt_w[val_idx], theta[val_idx],
                    y_mean, y_std, args.stage1_epochs, args.patience,
                    [(model.head.parameters(), args.stage1_lr)], anchor1, args.l2,
                    args.batch_size, args.seed + k, f"fold{k}-s1")
        state1 = {name: v.detach().clone() for name, v in model.state_dict().items()}

        if args.unfreeze == "all":
            for p in model.features.parameters():
                p.requires_grad = True
            feat_params = list(model.features.parameters())
        else:
            for p in model.features.parameters():
                p.requires_grad = False
            for p in model.features[9:].parameters():
                p.requires_grad = True
            feat_params = list(model.features[9:].parameters())
        for p in model.head.parameters():
            p.requires_grad = True
        anchor2 = capture_anchor(model)
        train_stage(model, X[train_idx], gt_n[train_idx], X[val_idx], gt_w[val_idx], theta[val_idx],
                    y_mean, y_std, args.stage2_epochs, args.patience,
                    [(feat_params, args.stage2_lr), (model.head.parameters(), args.stage2_lr_head)],
                    anchor2, args.l2, args.batch_size, args.seed + 100 + k, f"fold{k}-s2")

        pred_w = predict_world(model, X[test_idx], theta[test_idx], y_mean, y_std)
        oof_pred[test_idx] = pred_w
        test_metrics = spin_metrics(pred_w, gt_w[test_idx])
        base_test = spin_metrics(base_pred_w[test_idx], gt_w[test_idx])
        gt_rmse = eval_3dgt(model, gt_data, y_mean, y_std)
        fold_results.append({
            "fold": k,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "base_test": base_test,
            "finetuned_test": test_metrics,
            "base_3dgt_pos_cm": float(base_3dgt),
            "finetuned_3dgt_pos_cm": float(gt_rmse),
        })
        print(f"  test: base MAE {base_test['mae_overall']:.1f}/angle {base_test['angle_mean']:.1f} -> "
              f"finetuned MAE {test_metrics['mae_overall']:.1f}/angle {test_metrics['angle_mean']:.1f} | "
              f"3D_gt pos {base_3dgt:.2f} -> {gt_rmse:.2f} cm")
        torch.save({"state_dict": model.state_dict(), "norm": norm, "in_ch": 7, "L": L, "fold": k},
                   out_dir / f"fold_{k}.pt")

    def mean_std(key, sub):
        v = np.array([r[key][sub] for r in fold_results])
        return float(v.mean()), float(v.std())

    summary = {
        "n_usable": int(n),
        "folds": args.folds,
        "base_all": base_all,
        "base_3dgt_pos_cm": float(base_3dgt),
        "cv": {
            "base_mae_mean": mean_std("base_test", "mae_overall")[0],
            "base_mae_std": mean_std("base_test", "mae_overall")[1],
            "base_angle_mean": mean_std("base_test", "angle_mean")[0],
            "base_angle_std": mean_std("base_test", "angle_mean")[1],
            "finetuned_mae_mean": mean_std("finetuned_test", "mae_overall")[0],
            "finetuned_mae_std": mean_std("finetuned_test", "mae_overall")[1],
            "finetuned_angle_mean": mean_std("finetuned_test", "angle_mean")[0],
            "finetuned_angle_std": mean_std("finetuned_test", "angle_mean")[1],
            "finetuned_3dgt_mean": float(np.mean([r["finetuned_3dgt_pos_cm"] for r in fold_results])),
            "finetuned_3dgt_std": float(np.std([r["finetuned_3dgt_pos_cm"] for r in fold_results])),
        },
        "fold_results": fold_results,
    }
    with open(out_dir / "cv_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    with open(out_dir / "cv_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "n_train", "n_val", "n_test",
                    "base_mae", "base_angle", "ft_mae", "ft_angle", "base_3dgt_cm", "ft_3dgt_cm"])
        for r in fold_results:
            w.writerow([r["fold"], r["n_train"], r["n_val"], r["n_test"],
                        r["base_test"]["mae_overall"], r["base_test"]["angle_mean"],
                        r["finetuned_test"]["mae_overall"], r["finetuned_test"]["angle_mean"],
                        r["base_3dgt_pos_cm"], r["finetuned_3dgt_pos_cm"]])

    with open(out_dir / "oof_predictions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "pred_x", "pred_y", "pred_z", "gt_x", "gt_y", "gt_z"])
        for i in range(n):
            w.writerow([int(ids[i]), oof_pred[i, 0], oof_pred[i, 1], oof_pred[i, 2],
                        gt_w[i, 0], gt_w[i, 1], gt_w[i, 2]])

    print("\nCV summary")
    print(f"  base:      MAE {summary['cv']['base_mae_mean']:.1f}±{summary['cv']['base_mae_std']:.1f} | "
          f"angle {summary['cv']['base_angle_mean']:.1f}±{summary['cv']['base_angle_std']:.1f}")
    print(f"  finetuned: MAE {summary['cv']['finetuned_mae_mean']:.1f}±{summary['cv']['finetuned_mae_std']:.1f} | "
          f"angle {summary['cv']['finetuned_angle_mean']:.1f}±{summary['cv']['finetuned_angle_std']:.1f}")
    print(f"  3D_gt pos: base {base_3dgt:.2f} cm -> finetuned {summary['cv']['finetuned_3dgt_mean']:.2f}±{summary['cv']['finetuned_3dgt_std']:.2f} cm")
    print(f"  output: {out_dir}")

    all_state = None
    model_all = make_model(base_state)
    idx_all = np.arange(n)
    val_n = max(5, int(round(0.15 * n)))
    order = np.argsort(spins_norm)
    val_idx = order[:: max(1, n // val_n)][:val_n]
    train_idx = np.setdiff1d(idx_all, val_idx)
    for p in model_all.features.parameters():
        p.requires_grad = False
    for p in model_all.head.parameters():
        p.requires_grad = True
    anchor_all = capture_anchor(model_all)
    train_stage(model_all, X[train_idx], gt_n[train_idx], X[val_idx], gt_w[val_idx], theta[val_idx],
                y_mean, y_std, args.stage1_epochs, args.patience,
                [(model_all.head.parameters(), args.stage1_lr)], anchor_all, args.l2,
                args.batch_size, args.seed + 999, "all-s1")
    if args.unfreeze == "all":
        for p in model_all.features.parameters():
            p.requires_grad = True
        feat_params_all = list(model_all.features.parameters())
    else:
        for p in model_all.features.parameters():
            p.requires_grad = False
        for p in model_all.features[9:].parameters():
            p.requires_grad = True
        feat_params_all = list(model_all.features[9:].parameters())
    for p in model_all.head.parameters():
        p.requires_grad = True
    anchor_all2 = capture_anchor(model_all)
    train_stage(model_all, X[train_idx], gt_n[train_idx], X[val_idx], gt_w[val_idx], theta[val_idx],
                y_mean, y_std, args.stage2_epochs, args.patience,
                [(feat_params_all, args.stage2_lr), (model_all.head.parameters(), args.stage2_lr_head)],
                anchor_all2, args.l2, args.batch_size, args.seed + 1999, "all-s2")
    torch.save({"state_dict": model_all.state_dict(), "norm": norm, "in_ch": 7, "L": L,
                "trained_on": "trajectories", "n": n}, out_dir / "finetuned_trajectories.pt")
    print(f"saved final model: {out_dir / 'finetuned_trajectories.pt'}")


if __name__ == "__main__":
    main()
