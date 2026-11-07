import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def load_rows(meta_path):
    with open(meta_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_arrays(rows, traj_dir, L):
    X = np.zeros((len(rows), 7, L), dtype=np.float32)
    Yc = np.zeros((len(rows), 3), dtype=np.float32)
    Yo = np.zeros((len(rows), 3), dtype=np.float32)
    thetas = np.zeros(len(rows), dtype=np.float32)
    ids = np.zeros(len(rows), dtype=np.int64)
    grid = np.linspace(0.0, 1.0, L)
    for i, r in enumerate(rows):
        arr = np.loadtxt(str(traj_dir / f"sim_{int(r['id']):05d}.csv"), delimiter=",", skiprows=1)
        t = arr[:, 0].astype(np.float64)
        p = arr[:, 1:4].astype(np.float64)
        v = np.gradient(p, t, axis=0)
        omega = np.array([float(r["omega_x"]), float(r["omega_y"]), float(r["omega_z"])])
        theta = float(np.arctan2(v[0, 0], v[0, 1]))
        c = math.cos(theta)
        s = math.sin(theta)

        def rot(a):
            return np.column_stack([a[:, 0] * c - a[:, 1] * s, a[:, 0] * s + a[:, 1] * c, a[:, 2]])

        p_c = rot(p - p[0])
        v_c = rot(v)
        w_c = np.array([omega[0] * c - omega[1] * s, omega[0] * s + omega[1] * c, omega[2]])
        tau = (t - t[0]) / max(t[-1] - t[0], 1e-12)
        pr = np.column_stack([np.interp(grid, tau, p_c[:, j]) for j in range(3)])
        vr = np.column_stack([np.interp(grid, tau, v_c[:, j]) for j in range(3)])
        X[i, :3] = pr.T
        X[i, 3:6] = vr.T
        X[i, 6] = grid
        Yc[i] = w_c
        Yo[i] = omega
        thetas[i] = theta
        ids[i] = int(r["id"])
    return X, Yc, Yo, thetas, ids


def rotate_back(vec, theta):
    out = vec.copy()
    c = np.cos(theta)[:, None]
    s = np.sin(theta)[:, None]
    out[:, 0] = vec[:, 0] * c[:, 0] + vec[:, 1] * s[:, 0]
    out[:, 1] = -vec[:, 0] * s[:, 0] + vec[:, 1] * c[:, 0]
    return out


class SpinCNN(nn.Module):
    def __init__(self, in_ch=7, out=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, out),
        )

    def forward(self, x):
        h = self.features(x)
        h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.head(h)


def angle_error_deg(pred, true):
    pn = np.linalg.norm(pred, axis=1)
    tn = np.linalg.norm(true, axis=1)
    denom = np.maximum(pn * tn, 1e-12)
    cosv = np.clip(np.sum(pred * true, axis=1) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cosv))


def compute_metrics(pred, true):
    diff = pred - true
    out = {}
    for j, ax in enumerate("xyz"):
        out[f"mae_{ax}"] = float(np.mean(np.abs(diff[:, j])))
        out[f"rmse_{ax}"] = float(np.sqrt(np.mean(diff[:, j] ** 2)))
    out["mae_overall"] = float(np.mean(np.abs(diff)))
    out["rmse_overall"] = float(np.sqrt(np.mean(diff ** 2)))
    out["mag_mae"] = float(np.mean(np.abs(np.linalg.norm(pred, axis=1) - np.linalg.norm(true, axis=1))))
    ang = angle_error_deg(pred, true)
    out["angle_mean"] = float(np.mean(ang))
    out["angle_median"] = float(np.median(ang))
    out["angle_p95"] = float(np.percentile(ang, 95))
    return out


def run_epoch(model, loader, criterion, optimizer=None, noise_std=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            if training and noise_std is not None:
                xb = xb + torch.randn_like(xb) * noise_std
            if training:
                optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)
    return total / max(count, 1)


def predict(model, loader):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb).cpu().numpy())
            trues.append(yb.cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="prep_3d_gt/output/sim_pretrain/sim_meta.csv")
    ap.add_argument("--traj-dir", default="prep_3d_gt/output/sim_pretrain/trajectories")
    ap.add_argument("--out", default="pretrain/output")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    meta_path = Path(args.meta)
    if not meta_path.is_absolute():
        meta_path = root / meta_path
    traj_dir = Path(args.traj_dir)
    if not traj_dir.is_absolute():
        traj_dir = root / traj_dir
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    rows = load_rows(meta_path)
    splits = {"train": [], "val": [], "test": []}
    for r in rows:
        s = r.get("split", "train")
        if s in splits:
            splits[s].append(r)
    print({k: len(v) for k, v in splits.items()})

    print("building arrays")
    X_all, Yc_all, Yo_all, theta_all = {}, {}, {}, {}
    for name in ("train", "val", "test"):
        X, Yc, Yo, th, ids = build_arrays(splits[name], traj_dir, args.L)
        X_all[name] = X
        Yc_all[name] = Yc
        Yo_all[name] = Yo
        theta_all[name] = th
        print(f"  {name}: X {X.shape} Y {Yc.shape}")

    x_mean = X_all["train"].mean(axis=(0, 2), keepdims=True)
    x_std = X_all["train"].std(axis=(0, 2), keepdims=True) + 1e-6
    y_mean = Yc_all["train"].mean(axis=0)
    y_std = Yc_all["train"].std(axis=0) + 1e-6
    yo_mean = Yo_all["train"].mean(axis=0)

    norm = {
        "x_mean": x_mean.reshape(-1).tolist(),
        "x_std": x_std.reshape(-1).tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "L": args.L,
        "in_channels": 7,
    }
    with open(out_dir / "norm_stats.json", "w", encoding="utf-8") as fh:
        json.dump(norm, fh, indent=2)

    tensors = {}
    for name in ("train", "val", "test"):
        Xn = (X_all[name] - x_mean) / x_std
        Yn = (Yc_all[name] - y_mean) / y_std
        tensors[name] = (
            torch.from_numpy(Xn.astype(np.float32)),
            torch.from_numpy(Yn.astype(np.float32)),
        )

    train_ds = TensorDataset(tensors["train"][0], tensors["train"][1])
    val_ds = TensorDataset(tensors["val"][0], tensors["val"][1])
    test_ds = TensorDataset(tensors["test"][0], tensors["test"][1])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    noise_phys = np.array([0.004, 0.004, 0.004, 0.04, 0.04, 0.04, 0.0], dtype=np.float32)
    noise_std = torch.from_numpy(noise_phys / x_std.reshape(-1)).view(1, 7, 1)

    model = SpinCNN(in_ch=7, out=3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    wait = 0
    log = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, noise_std=noise_std)
        val_loss = run_epoch(model, val_loader, criterion, None)
        val_pred_n, _ = predict(model, val_loader)
        val_pred_c = val_pred_n * y_std + y_mean
        val_pred = rotate_back(val_pred_c, theta_all["val"])
        val_true = Yo_all["val"]
        vm = compute_metrics(val_pred, val_true)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": vm["mae_overall"],
            "val_angle_mean": vm["angle_mean"],
            "lr": lr_now,
        })
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:3d} | train {train_loss:.5f} | val {val_loss:.5f} | "
                f"mae {vm['mae_overall']:.2f} rad/s | angle {vm['angle_mean']:.2f} deg | lr {lr_now:.1e}"
            )
        if wait >= args.patience:
            print(f"early stop at epoch {epoch}, best epoch {best_epoch}")
            break

    model.load_state_dict(best_state)
    with open(out_dir / "train_log.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "val_loss", "val_mae", "val_angle_mean", "lr"])
        w.writeheader()
        for row in log:
            w.writerow(row)

    test_pred_n, _ = predict(model, test_loader)
    test_pred_c = test_pred_n * y_std + y_mean
    test_pred = rotate_back(test_pred_c, theta_all["test"])
    test_true = Yo_all["test"]
    test_metrics = compute_metrics(test_pred, test_true)
    val_pred_n, _ = predict(model, val_loader)
    val_pred_c = val_pred_n * y_std + y_mean
    val_pred = rotate_back(val_pred_c, theta_all["val"])
    val_true = Yo_all["val"]
    val_metrics = compute_metrics(val_pred, val_true)
    baseline_true = test_true
    baseline_pred = np.tile(yo_mean, (len(test_true), 1))
    baseline_metrics = compute_metrics(baseline_pred, baseline_true)

    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "n_params": n_params,
        "val": val_metrics,
        "test": test_metrics,
        "baseline_test": baseline_metrics,
        "norm": norm,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    torch.save({"state_dict": model.state_dict(), "norm": norm, "in_ch": 7, "L": args.L}, out_dir / "best_model.pt")

    with open(out_dir / "test_predictions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pred_x", "pred_y", "pred_z", "true_x", "true_y", "true_z"])
        for p, t in zip(test_pred, test_true):
            w.writerow([p[0], p[1], p[2], t[0], t[1], t[2]])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [r["epoch"] for r in log]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(epochs, [r["train_loss"] for r in log], label="train")
        axes[0].plot(epochs, [r["val_loss"] for r in log], label="val")
        axes[0].set_title("loss")
        axes[0].legend()
        axes[1].plot(epochs, [r["val_mae"] for r in log])
        axes[1].set_title("val MAE (rad/s)")
        axes[2].plot(epochs, [r["val_angle_mean"] for r in log])
        axes[2].set_title("val angle error (deg)")
        fig.tight_layout()
        fig.savefig(out_dir / "training_curves.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"plot skipped: {exc}")

    print("\ntest metrics")
    print(f"  MAE overall: {test_metrics['mae_overall']:.2f} rad/s")
    print(f"  RMSE overall: {test_metrics['rmse_overall']:.2f} rad/s")
    print(f"  |w| MAE: {test_metrics['mag_mae']:.2f} rad/s")
    print(f"  angle mean/median/p95: {test_metrics['angle_mean']:.2f} / {test_metrics['angle_median']:.2f} / {test_metrics['angle_p95']:.2f} deg")
    print(f"  baseline MAE: {baseline_metrics['mae_overall']:.2f} rad/s, angle {baseline_metrics['angle_mean']:.2f} deg")
    print(f"  output: {out_dir}")


if __name__ == "__main__":
    main()
