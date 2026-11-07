# SpinBench-PP

基于物理信息神经网络（PINN / physics-informed）的乒乓球轨迹建模与自旋估计项目。

项目以乒乓球飞行段的物理方程为核心约束，用 3D 轨迹训练/微调模型，估计自旋向量，并支持轨迹重建、落点/过网分析、轨迹拟合对比和自旋可辨识性分析。

---

## 1. 项目简介

乒乓球的自旋会通过马格努斯力显著改变飞行轨迹，但视觉系统通常很难直接测量自旋。本项目把飞行段物理方程与数据驱动模型结合：

- 输入：一条 3D 轨迹（`t, x, y, z`）；
- 输出：自旋向量 `ω = (ωx, ωy, ωz)`；
- 结合物理 ODE 可进一步做：
  - 轨迹重建/去噪；
  - 未来位置预测；
  - 第一次落点、过网高度估计；
  - 轨迹拟合方法对比；
  - 自旋可辨识性（`ω⊥` / `ω∥`）分析。

项目采用 physics-informed 方式：物理方程用于仿真数据生成、rollout 自监督损失和轨迹重建；网络本身是一个 1D CNN，不是硬编码 PDE 的经典 PINN。

---

## 2. 背景与目标

- 传统数值方法依赖精确初始条件和物理参数；
- 纯数据模型在数据稀疏时容易给出非物理结果；
- 本项目用飞行段 ODE 作为先验，联合数据与物理约束，从轨迹中估计自旋并重建轨迹；
- 同时提供多组实验脚本，用于比较多项式拟合、物理拟合、Tebbe 线性法和神经网络模型。

---

## 3. 核心方法

### 3.1 物理模型

只建模第一次弹跳前的飞行段，忽略弹跳、旋转衰减等。

飞行段 ODE：

```
a = g − (KD / M) · ||v|| · v + (KM / M) · (ω × v)
```

- `g`：重力加速度；
- `KD`：阻力系数（与质量归一化前）；
- `KM`：马格努斯系数；
- `M`：球质量；
- `v`：速度；
- `ω`：自旋角速度。

其中马格努斯力只依赖 `ω × v`，这决定了轨迹能约束的是自旋的垂直分量 `ω⊥`；沿速度方向的 `ω∥` 不产生力，无法由轨迹唯一确定。

### 3.2 数据与规范化

模型输入统一为 64 点轨迹：

1. 截断到第一次弹跳前；
2. 重采样到固定 64 点（归一化时间的线性插值）；
3. 平移到首点；
4. 绕 Z 轴旋转，使初始水平速度对齐 +Y；
5. 用预训练统计量做通道归一化；
6. 输入通道：`x, y, z, vx, vy, vz, tau`。

速度由原始时间戳求导得到；`tau` 为归一化时间。

### 3.3 模型

- 1D CNN；
- 输入 `64 × 7`；
- 全局平均/最大池化 + MLP 输出 3 维自旋；
- 支持从仿真预训练模型或 3D_gt 微调模型继续训练。

### 3.4 训练与微调

- **预训练**：用仿真轨迹 + GT 自旋做监督学习；
- **3D_gt 微调**：无 spin 标签，使用物理 rollout 位置损失做自监督；
- **trajectories 微调**：可选监督损失/物理损失，支持 5 折交叉验证；
- **physics-only 微调**：只用轨迹 + 物理 rollout，不使用 `index.csv`。

### 3.5 轨迹重建与自旋评估

- 用模型预测自旋；
- 用飞行 ODE 从首点 + 初始速度积分，重建 64 点或原始时间戳上的轨迹；
- 评估：
  - 位置重建 RMSE；
  - 完整 `ω` 与 `ω⊥` 的 MAE/角度；
  - 物理残差、条件数、可辨识性等。

---

## 4. 数据说明

项目包含两组输入数据：

- `3D_gt/`
  - 真实 3D 轨迹；
  - 格式：`Timestamp, X, Y, Z`；
  - 单位：m、s；
  - 坐标系：X 沿球台宽度、Y 沿球台长度、Z 向上，原点在台面中心；
  - 无自旋标签。

- `trajectories/`
  - 轨迹数据 + `index.csv` 自旋标签；
  - 格式：`t; x; y; z`（分号分隔）；
  - `index.csv`：`traj_file, x_spin, y_spin, z_spin`；
  - 标签包含噪声/离群，主要用于评估与对照。

---

## 5. 目录结构

```
3D_gt/                     原始 3D_gt 轨迹
trajectories/              轨迹 + index.csv 自旋标签

prep_3d_gt/                3D_gt 数据准备与仿真数据生成
  prepare_3d_gt.py
  generate_sim.py
  output/

pretrain/                  模型训练、微调、拟合对比、分析脚本
  train.py
  finetune.py
  finetune_trajectories.py
  fit_compare.py
  fit_compare_raw.py
  fit_trajectories_physics.py
  tebbe_fit.py
  physics_finetune.py
  plot_xy.py
  plot_straightness.py
  output/
  timing/
```

---

## 6. 环境与依赖

推荐使用项目内虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy scipy matplotlib
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

- Python 3.10；
- PyTorch CPU；
- numpy / scipy / matplotlib；
- Excel 导出脚本使用全局 Python 的 pandas / openpyxl。

---

## 7. 脚本说明

| 脚本 | 作用 |
|---|---|
| `prep_3d_gt/prepare_3d_gt.py` | 3D_gt 审计、清洗、截断、重采样、伪 spin 拟合、划分 |
| `prep_3d_gt/generate_sim.py` | 用 3D_gt 种子 + LHS 生成仿真预训练数据 |
| `pretrain/train.py` | 1D CNN 预训练 |
| `pretrain/finetune.py` | 在 3D_gt 上做物理自监督微调 |
| `pretrain/finetune_trajectories.py` | 在 trajectories 上做监督微调 / 5 折 CV |
| `pretrain/fit_compare.py` | 二次/三次多项式 vs 模型（64 点网格） |
| `pretrain/fit_compare_raw.py` | 二次/三次多项式 vs 模型（原始点） |
| `pretrain/fit_trajectories_physics.py` | trajectories 上的物理 shooting fit |
| `pretrain/tebbe_fit.py` | Tebbe 式多时刻线性最小二乘 + 条件数分析 |
| `pretrain/physics_finetune.py` | physics-only 5 折微调 |
| `pretrain/plot_xy.py` | 三方法 XY 投影对比图 |
| `pretrain/plot_straightness.py` | trajectories 与 3D_gt 的 XY 直线性对比 |

---

## 8. 输出说明

### 8.1 `prep_3d_gt/output/`

| 路径 | 内容 |
|---|---|
| `3D_gt_clean/` | 清洗、截断、重采样后的 3D_gt 轨迹（`Timestamp,X,Y,Z`） |
| `3D_gt_meta.csv` | 每条轨迹的伪 spin、v0、残差、点数、划分等元数据 |
| `3D_gt_splits.csv` | train/val/test 划分 |
| `audit_report.json` | 数据审计：采样间隔、范围、异常、flag 等 |
| `calibration.json` | KD 校准结果与说明 |
| `sim_pretrain/` | 仿真预训练数据 |
| `sim_pretrain/trajectories/` | 仿真轨迹 CSV（与 3D_gt 同格式） |
| `sim_pretrain/sim_meta.csv` | 仿真轨迹的 GT spin、初始条件、落点、划分 |
| `sim_pretrain/sim_splits.csv` | 仿真数据划分 |
| `sim_pretrain/coverage.png` | 仿真 vs 3D_gt 分布覆盖图 |
| `sim_pretrain/report.json` | 生成数量、接受率、覆盖率、KS 统计等 |

### 8.2 `pretrain/output/` 顶层文件

| 路径 | 内容 |
|---|---|
| `best_model.pt` | 仿真预训练模型 |
| `norm_stats.json` | 输入/自旋归一化统计 |
| `metrics.json` | 预训练评估指标 |
| `train_log.csv` | 预训练日志 |
| `training_curves.png` | 预训练曲线 |
| `test_predictions.csv` | 预训练测试集预测 |
| `finetuned_model.pt` | 3D_gt 微调模型（顶层） |
| `finetune_metrics.json` | 3D_gt 微调指标 |
| `finetune_log.csv` | 3D_gt 微调日志 |
| `finetune_curves.png` | 3D_gt 微调曲线 |
| `finetuned_predictions.csv` | 3D_gt 微调预测 |
| `3d_gt_length_curvature.csv` | 3D_gt 轨迹长度/弯曲度统计 |

### 8.3 `pretrain/output/` 子目录

| 路径 | 内容 |
|---|---|
| `finetune/` | 3D_gt 物理自监督微调：模型、指标、日志、曲线、预测 |
| `fit_compare/` | 64 点网格上的三方法对比：`fit_compare.csv/json/png`、`xy/xy_001~010.png`、`xy/xy_all.png` |
| `fit_compare_raw/` | 原始点上的三方法对比：`fit_compare_raw.csv/json/png`、`compare_raw_vs_resampled.png`、`xy/xy_001~010.png`、`xy/xy_all.png` |
| `fit_compare_raw/xy_001_data.xlsx` | id 001 的 points / smooth_xy / rmse / raw_full / meta |
| `fit_compare_raw/xy_001_data_with_dt.xlsx` | 在上一份基础上增加 dt 间隔与直方图 |
| `fit_compare_raw/xy_001_dt_hist.png` | id 001 原始时间间隔直方图 |
| `fit_compare_raw/points_excel/` | id 015、id 124 的 `points` Excel |
| `fit_compare_raw/points_excel/id_015_points.xlsx` | id 015 原始点 + 三方法拟合点 |
| `fit_compare_raw/points_excel/id_124_points.xlsx` | id 124 原始点 + 三方法拟合点 |
| `straightness/` | trajectories 前 10 条合格轨迹的 XY 投影与直线性对比 |
| `straightness/trajectories_xy/` | 这 10 条轨迹的单独 XY 图 |
| `straightness/trajectories_all.png` | 10 条 trajectories 的 2×5 汇总 |
| `straightness/3dgt_all.png` | 3D_gt 前 10 条的 2×5 汇总 |
| `straightness/compare_3dgt_vs_trajectories.png` | 两组数据的等比例 XY 对比 |
| `straightness/straightness_deviation.png` | 归一化横向偏移 vs 沿弦位置 |
| `fit_trajectories_physics/` | trajectories 物理 shooting fit：`fit_results.csv/json`、`fit_vs_index.png` |
| `tebbe_fit/` | Tebbe 式线性拟合：`tebbe_results.csv`、`tebbe_summary.json`、`tebbe_analysis.png` |
| `physics_finetune/` | physics-only 5 折微调：`physics_finetune_summary.json` |
| `physics_finetune/finetune_finetuned_model/` | 从 3D_gt 模型初始化：`cv_metrics.csv/json`、`oof_predictions.csv` |
| `physics_finetune/output_best_model/` | 从仿真模型初始化：`cv_metrics.csv/json`、`oof_predictions.csv` |
| `finetune_trajectories/` | trajectories 监督微调（方案 A/B）：`cv_metrics.csv/json`、`finetuned_trajectories.pt`、`fold_0~4.pt`、`oof_predictions.csv` |
| `finetune_trajectories_planB/` | 方案 B（无 anchor）输出 |
| `finetune_trajectories_B1/` | 方案 B1（全部卷积层）输出 |
| `finetune_trajectories_B2/` | 方案 B2（从仿真模型初始化）输出 |
| `finetune_trajectories_smoke/` | 上述流程的临时 smoke 输出 |
| `fit_trajectories_physics_smoke/` | 物理拟合的临时 smoke 输出 |
| `fit_trajectories_physics_smoke10/` | 物理拟合前 10 条的临时输出 |
| `physics_finetune_smoke/` | physics-only 微调的临时 smoke 输出 |
| `3d_gt_longest/` | 最长/最弯轨迹图：`id_015.png`、`id_124_xy.png` |

### 8.4 `pretrain/timing/`

预训练计时用的临时基准输出：

- `best_model.pt`
- `metrics.json`
- `norm_stats.json`
- `test_predictions.csv`
- `training_curves.png`
- `train_log.csv`

---

## 9. 复现流程

### 9.1 创建环境

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy scipy matplotlib
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 9.2 3D_gt 数据准备

```powershell
.\.venv\Scripts\python.exe prep_3d_gt\prepare_3d_gt.py
```

### 9.3 生成仿真预训练数据

```powershell
.\.venv\Scripts\python.exe prep_3d_gt\generate_sim.py
```

### 9.4 预训练

```powershell
.\.venv\Scripts\python.exe pretrain\train.py --epochs 200 --patience 40
```

### 9.5 3D_gt 微调

```powershell
.\.venv\Scripts\python.exe pretrain\finetune.py --stage1-epochs 40 --stage2-epochs 40 --patience 12
```

### 9.6 轨迹拟合对比

```powershell
.\.venv\Scripts\python.exe pretrain\fit_compare.py
.\.venv\Scripts\python.exe pretrain\fit_compare_raw.py
```

### 9.7 物理拟合与 Tebbe 分析

```powershell
.\.venv\Scripts\python.exe pretrain\fit_trajectories_physics.py
.\.venv\Scripts\python.exe pretrain\tebbe_fit.py
.\.venv\Scripts\python.exe pretrain\physics_finetune.py
```

### 9.8 trajectories 微调

```powershell
.\.venv\Scripts\python.exe pretrain\finetune_trajectories.py --out pretrain\output\finetune_trajectories
```

### 9.9 可视化

```powershell
.\.venv\Scripts\python.exe pretrain\plot_xy.py
.\.venv\Scripts\python.exe pretrain\plot_straightness.py
```

---

## 10. 使用说明

- 本项目只建模**第一次弹跳前**的飞行段，不包含弹跳、旋转衰减等；
- 坐标系为球台中心系：X 宽、Y 长、Z 上；
- 所有位置单位为 m，时间为 s，自旋为 rad/s，速度为 m/s；
- 轨迹只能约束 `ω × v`，因此完整自旋的沿速度分量不可唯一确定；实际评估建议关注 `ω⊥` 和轨迹重建误差；
- KD/KM 固定时，自旋幅值与 KM 的标定相关；
- `*_smoke` 与 `timing` 目录为临时/基准输出，不属于正式结果。

---

## 11. 参考

- Physics-Informed Neural Networks（PINN）相关文献；
- J. Tebbe, L. Klamt, Y. Gao, A. Zell, *Spin Detection in Robotic Table Tennis*, ICRA 2020；
- tt3d / 3D_gt 数据说明。
