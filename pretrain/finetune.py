import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import SpinCNN

G_NP = np.array([0.0, 0.0, -9.81], dtype=np.float64)
G_T = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32)
R = 0.02
M = 2.7e-3
KD = 3.9827679117703966e-4
KM = 4.86e-6


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_rows(meta_path):
    with open(meta_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def rotate_xy(a, theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return np.column_stack([a[:, 0] * c - a[:, 1] * s, a[:, 0] * s + a[:, 1] * c, a[:, 2]])


def rotate_back_np(v, theta):
    c = np.cos(theta)[:, None]
    s = np.sin(theta)[:, None]
    out = v.copy()
    out[:, 0] = v[:, 0] * c[:, 0] + v[:, 1] * s[:, 0]
    out[:, 1] = -v[:, 0] * s[:, 0] + v[:, 1] * c[:, 0]
    return out


def rotate_back_t(v, theta):
    c = torch.cos(theta).unsqueeze(1)
    s = torch.sin(theta).unsqueeze(1)
    out = v.clone()
    out[:, 0] = v[:, 0] * c[:, 0] + v[:, 1] * s[:, 0]
    out[:, 1] = -v[:, 0] * s[:, 0] + v[:, 1] * c[:, 0]
    return out


def build_split(rows, clean_dir, L, norm):
    n = len(rows)
    X = np.zeros((n, 7, L), dtype=np.float32)
    pos = np.zeros((n, L, 3), dtype=np.float32)
    vel = np.zeros((n, L, 3), dtype=np.float32)
    acc = np.zeros((n, L, 3), dtype=np.float32)
    dts = np.zeros(n, dtype=np.float32)
    thetas = np.zeros(n, dtype=np.float32)
    pseudo = np.zeros((n, 3), dtype=np.float32)
    v0_fit = np.zeros((n, 3), dtype=np.float32)
    ids = np.zeros(n, dtype=np.int64)
    grid = np.linspace(0.0, 1.0, L)
    for i, r in enumerate(rows):
        arr = np.loadtxt(str(clean_dir / f"{int(r['id']):03d}.csv"), delimiter=",", skiprows=1)
        t = arr[:, 0].astype(np.float64)
        p = arr[:, 1:4].astype(np.float64)
        T = t[-1] - t[0]
        dt = float(np.median(np.diff(t)))
        v = np.gradient(p, t, axis=0)
        theta = float(np.arctan2(v[0, 0], v[0, 1]))
        p_c = rotate_xy(p - p[0], theta)
        v_c = rotate_xy(v, theta)
        npts = len(p_c)
        wa = min(11, npts if npts % 2 == 1 else npts - 1)
        if wa % 2 == 0:
            wa -= 1
        if wa >= 5:
            a_c = np.column_stack([
                savgol_filter(p_c[:, j], wa, 2, deriv=2, delta=dt, mode="interp") for j in range(3)
            ])
        else:
            a_c = np.gradient(v_c, t, axis=0)
        tau = (t - t[0]) / max(T, 1e-12)
        p_r = np.column_stack([np.interp(grid, tau, p_c[:, j]) for j in range(3)])
        v_r = np.column_stack([np.interp(grid, tau, v_c[:, j]) for j in range(3)])
        a_r = np.column_stack([np.interp(grid, tau, a_c[:, j]) for j in range(3)])
        X[i, :3] = p_r.T
        X[i, 3:6] = v_r.T
        X[i, 6] = grid
        pos[i] = p_r
        vel[i] = v_r
        acc[i] = a_r
        dts[i] = T / (L - 1)
        thetas[i] = theta
        pseudo[i] = [float(r["omega_x"]), float(r["omega_y"]), float(r["omega_z"])]
        v0w = np.array([float(r["v0_x"]), float(r["v0_y"]), float(r["v0_z"])], dtype=np.float64)
        v0_fit[i] = rotate_xy(v0w[None, :], theta)[0]
        ids[i] = int(r["id"])
    x_mean = np.array(norm["x_mean"], dtype=np.float32).reshape(1, 7, 1)
    x_std = np.array(norm["x_std"], dtype=np.float32).reshape(1, 7, 1)
    Xn = (X - x_mean) / x_std
    return {
        "X": torch.from_numpy(Xn.astype(np.float32)),
        "pos": torch.from_numpy(pos),
        "vel": torch.from_numpy(vel),
        "acc": torch.from_numpy(acc),
        "dts": torch.from_numpy(dts),
        "theta": torch.from_numpy(thetas),
        "pseudo": torch.from_numpy(pseudo),
        "v0_fit": torch.from_numpy(v0_fit),
        "ids": ids,
    }


def accel_vec(v, w, kd, km):
    speed = torch.linalg.norm(v, dim=1, keepdim=True)
    drag = -(kd / M) * speed * v
    magnus = (km / M) * torch.cross(w, v, dim=1)
    return G_T + drag + magnus


def accel_field(v, w, kd, km):
    speed = torch.linalg.norm(v, dim=2, keepdim=True)
    drag = -(kd / M) * speed * v
    wb = w.unsqueeze(1).expand_as(v)
    magnus = (km / M) * torch.cross(wb, v, dim=2)
    return G_T + drag + magnus


def rollout(v0, w, dts, L, kd, km):
    v = v0.clone()
    r = torch.zeros_like(v0)
    out = [r]
    for _ in range(L - 1):
        dt = dts.view(-1, 1)
        k1v = accel_vec(v, w, kd, km)
        k1r = v
        k2v = accel_vec(v + 0.5 * dt * k1v, w, kd, km)
        k2r = v + 0.5 * dt * k1v
        k3v = accel_vec(v + 0.5 * dt * k2v, w, kd, km)
        k3r = v + 0.5 * dt * k2v
        k4v = accel_vec(v + dt * k3v, w, kd, km)
        k4r = v + dt * k3v
        v = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        r = r + dt / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
        out.append(r)
    return torch.stack(out, dim=1)


def forward_loss(model, batch, y_mean, y_std, kd, km, phys_weight, reg_weight):
    X, pos, vel, acc, dts, v0 = batch
    pred_n = model(X)
    w = pred_n * y_std + y_mean
    w = torch.clamp(w, -1000.0, 1000.0)
    pred_pos = rollout(v0, w, dts, pos.shape[1], kd, km)
    pos_loss = F.huber_loss(pred_pos, pos, delta=0.02)
    a_pred = accel_field(vel, w, kd, km)
    phys_loss = F.mse_loss(a_pred, acc)
    reg = (w ** 2).mean()
    loss = pos_loss + phys_weight * phys_loss / 100.0 + reg_weight * reg
    return loss, pos_loss, phys_loss, pred_pos, w


def evaluate(model, loader, y_mean, y_std, kd, km, thetas, pseudo, ids, split_name):
    model.eval()
    preds_pos = []
    preds_w = []
    obs_pos = []
    phys_err = []
    with torch.no_grad():
        for X, pos, vel, acc, dts, v0 in loader:
            pred_n = model(X)
            w = pred_n * y_std + y_mean
            w = torch.clamp(w, -1000.0, 1000.0)
            rp = rollout(v0, w, dts, pos.shape[1], kd, km)
            a_pred = accel_field(vel, w, kd, km)
            preds_pos.append(rp)
            preds_w.append(w)
            obs_pos.append(pos)
            phys_err.append((a_pred - acc).pow(2).sum(dim=2))
    pred_pos = torch.cat(preds_pos, dim=0)
    obs = torch.cat(obs_pos, dim=0)
    w = torch.cat(preds_w, dim=0)
    phys = torch.cat(phys_err, dim=0)
    pos_rmse = torch.sqrt(((pred_pos - obs) ** 2).sum(dim=2)).mean().item() * 100.0
    phys_rmse = torch.sqrt(phys.mean()).item()
    w_world = rotate_back_t(w, thetas)
    diff = w_world - pseudo
    mae = diff.abs().mean().item()
    pn = torch.linalg.norm(w_world, dim=1)
    tn = torch.linalg.norm(pseudo, dim=1)
    cosv = torch.clamp((w_world * pseudo).sum(dim=1) / torch.clamp(pn * tn, min=1e-12), -1.0, 1.0)
    ang = torch.rad2deg(torch.arccos(cosv))
    metrics = {
        "split": split_name,
        "n": int(len(obs)),
        "pos_rmse_cm": float(pos_rmse),
        "phys_rmse": float(phys_rmse),
        "pseudo_mae": float(mae),
        "pseudo_angle_mean": float(ang.mean().item()),
        "pseudo_angle_median": float(ang.median().item()),
        "pseudo_angle_p95": float(torch.quantile(ang, 0.95).item()),
        "omega_norm_median": float(pn.median().item()),
        "omega_norm_p95": float(torch.quantile(pn, 0.95).item()),
    }
    return metrics, w_world.numpy(), pseudo.numpy(), ids


def make_loader(data, batch_size, shuffle):
    ds = TensorDataset(data["X"], data["pos"], data["vel"], data["acc"], data["dts"], data["v0_fit"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_stage(model, train_loader, val_loader, y_mean, y_std, kd, km, epochs, patience, lr, phys_weight, reg_weight, tag, log, val_theta, val_pseudo, val_ids):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-3)
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = -1
    wait = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, pos_loss, phys_loss, _, _ = forward_loss(
                model, batch, y_mean, y_std, kd, km, phys_weight, reg_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total += float(loss.item()) * len(batch[0])
            count += len(batch[0])
        train_loss = total / max(count, 1)
        val_metrics, _, _, _ = evaluate(model, val_loader, y_mean, y_std, kd, km, val_theta, val_pseudo, val_ids, "val")
        val_rmse = val_metrics["pos_rmse_cm"]
        log.append({
            "stage": tag,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_pos_rmse_cm": val_rmse,
            "val_phys_rmse": val_metrics["phys_rmse"],
            "val_pseudo_angle_mean": val_metrics["pseudo_angle_mean"],
        })
        if val_rmse < best_val - 1e-6:
            best_val = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  [{tag}] epoch {epoch:3d} | loss {train_loss:.6f} | val pos {val_rmse:.2f} cm | "
                  f"phys {val_metrics['phys_rmse']:.2f} | pseudo angle {val_metrics['pseudo_angle_mean']:.2f}")
        if wait >= patience:
            print(f"  [{tag}] early stop at {epoch}, best epoch {best_epoch}")
            break
    model.load_state_dict(best_state)
    return best_val, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="prep_3d_gt/output/3D_gt_meta.csv")
    ap.add_argument("--clean-dir", default="prep_3d_gt/output/3D_gt_clean")
    ap.add_argument("--checkpoint", default="pretrain/output/best_model.pt")
    ap.add_argument("--out", default="pretrain/output/finetune")
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage1-epochs", type=int, default=40)
    ap.add_argument("--stage2-epochs", type=int, default=40)
    ap.add_argument("--stage1-lr", type=float, default=1e-3)
    ap.add_argument("--stage2-lr", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--phys-weight", type=float, default=1e-3)
    ap.add_argument("--reg-weight", type=float, default=1e-9)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    def rel(p):
        q = Path(p)
        return q if q.is_absolute() else root / q
    meta_path = rel(args.meta)
    clean_dir = rel(args.clean_dir)
    ckpt_path = rel(args.checkpoint)
    out_dir = rel(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    norm = ckpt["norm"]
    model = SpinCNN(in_ch=7, out=3)
    model.load_state_dict(ckpt["state_dict"])
    y_mean = torch.tensor(norm["y_mean"], dtype=torch.float32).view(1, 3)
    y_std = torch.tensor(norm["y_std"], dtype=torch.float32).view(1, 3)

    rows = [r for r in load_rows(meta_path) if int(r["unusable"]) == 0]
    split_rows = {"train": [], "val": [], "test": []}
    for r in rows:
        s = r.get("split", "train")
        if s in split_rows:
            split_rows[s].append(r)
    print({k: len(v) for k, v in split_rows.items()})

    data = {}
    for name in ("train", "val", "test"):
        data[name] = build_split(split_rows[name], clean_dir, args.L, norm)
        print(f"  {name}: {len(split_rows[name])}")

    train_loader = make_loader(data["train"], args.batch_size, True)
    val_loader = make_loader(data["val"], args.batch_size, False)
    test_loader = make_loader(data["test"], args.batch_size, False)

    def full_eval(loader, split_name, key):
        return evaluate(model, loader, y_mean, y_std, KD, KM,
                        data[key]["theta"], data[key]["pseudo"], data[key]["ids"], split_name)

    pre_val, _, _, _ = full_eval(val_loader, "val", "val")
    pre_test, pre_w, pre_true, pre_ids = full_eval(test_loader, "test", "test")
    print(f"pretrained zero-shot: val pos {pre_val['pos_rmse_cm']:.2f} cm | test {pre_test['pos_rmse_cm']:.2f} cm | "
          f"pseudo angle {pre_test['pseudo_angle_mean']:.2f}")

    log = []

    print("stage 1: freeze features, train head")
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    s1_val, s1_epoch = train_stage(model, train_loader, val_loader, y_mean, y_std, KD, KM,
                                   args.stage1_epochs, args.patience, args.stage1_lr,
                                   args.phys_weight, args.reg_weight, "stage1", log,
                                   data["val"]["theta"], data["val"]["pseudo"], data["val"]["ids"])
    s1_val_metrics, _, _, _ = full_eval(val_loader, "val", "val")
    s1_test, s1_w, s1_true, s1_ids = full_eval(test_loader, "test", "test")
    state1 = {k: v.detach().clone() for k, v in model.state_dict().items()}

    print("stage 2: unfreeze last conv block")
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.features[9:].parameters():
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
    s2_val, s2_epoch = train_stage(model, train_loader, val_loader, y_mean, y_std, KD, KM,
                                   args.stage2_epochs, args.patience, args.stage2_lr,
                                   args.phys_weight, args.reg_weight, "stage2", log,
                                   data["val"]["theta"], data["val"]["pseudo"], data["val"]["ids"])
    s2_val_metrics, _, _, _ = full_eval(val_loader, "val", "val")
    s2_test, s2_w, s2_true, s2_ids = full_eval(test_loader, "test", "test")
    state2 = {k: v.detach().clone() for k, v in model.state_dict().items()}

    candidates = [
        ("pretrained", pre_val["pos_rmse_cm"], pre_test, None),
        ("stage1", s1_val_metrics["pos_rmse_cm"], s1_test, None),
        ("stage2", s2_val_metrics["pos_rmse_cm"], s2_test, None),
    ]
    best_name = min(candidates, key=lambda x: x[1])[0]
    print(f"best by val pos rmse: {best_name}")

    pred_map = {
        "pretrained": (pre_w, pre_true, pre_ids),
        "stage1": (s1_w, s1_true, s1_ids),
        "stage2": (s2_w, s2_true, s2_ids),
    }
    if best_name == "pretrained":
        model.load_state_dict(ckpt["state_dict"])
    elif best_name == "stage1":
        model.load_state_dict(state1)
    else:
        model.load_state_dict(state2)
    best_w, best_true, best_ids = pred_map[best_name]

    torch.save({"state_dict": model.state_dict(), "norm": norm, "in_ch": 7, "L": args.L, "stage": best_name},
               out_dir / "finetuned_model.pt")

    metrics = {
        "best": best_name,
        "pretrained": {"val": pre_val, "test": pre_test},
        "stage1": {"val": s1_val_metrics, "test": s1_test, "best_epoch": s1_epoch},
        "stage2": {"val": s2_val_metrics, "test": s2_test, "best_epoch": s2_epoch},
        "kd": KD,
        "km": KM,
    }
    with open(out_dir / "finetune_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    with open(out_dir / "finetune_log.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["stage", "epoch", "train_loss", "val_pos_rmse_cm", "val_phys_rmse", "val_pseudo_angle_mean"])
        w.writeheader()
        for row in log:
            w.writerow(row)

    with open(out_dir / "test_predictions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "pred_x", "pred_y", "pred_z", "pseudo_x", "pseudo_y", "pseudo_z"])
        for i, p, t in zip(best_ids, best_w, best_true):
            w.writerow([int(i), p[0], p[1], p[2], t[0], t[1], t[2]])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for stage in ("stage1", "stage2"):
            sub = [r for r in log if r["stage"] == stage]
            if sub:
                xs = [r["epoch"] for r in sub]
                axes[0].plot(xs, [r["train_loss"] for r in sub], label=f"{stage} loss")
                axes[1].plot(xs, [r["val_pos_rmse_cm"] for r in sub], label=f"{stage} val pos")
        axes[0].set_title("train loss")
        axes[0].legend()
        axes[1].set_title("val position RMSE (cm)")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(out_dir / "finetune_curves.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"plot skipped: {exc}")

    print("\nsummary")
    for name, val_rmse, test, _ in candidates:
        print(f"  {name}: val pos {val_rmse:.2f} cm | test pos {test['pos_rmse_cm']:.2f} cm | "
              f"pseudo angle {test['pseudo_angle_mean']:.2f} | phys {test['phys_rmse']:.2f}")
    print(f"  output: {out_dir}")


if __name__ == "__main__":
    main()
