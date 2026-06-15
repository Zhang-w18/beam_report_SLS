# Sionna SLS Beam Management Platform v2.4

本项目是一个面向“服务波束 + 干扰波束上报”的系统级波束管理仿真原型。默认场景为 **1-site 3-sector**，默认 TRP 天线为：

4 TXRUs, 1024 AEs：

$$
(M, N, P, M_g, N_g; M_p, N_p) = (16, 16, 2, 2, 1; 1, 1)
$$

$$
(d_H, d_V) = (0.5, 0.5)
$$

v2.4 的核心变化是新增了 **RF architecture** 配置层，代码会自动把射频架构、波束发射方式和 MU order 关联起来：

- 情况 1：`panel_polarization_subarray`，即 sub-connected / panel-polarization connected；
- 情况 2：`fully_connected`，即 fully-connected hybrid beamforming；
- 默认参数为情况 1，允许不同极化采用不同波束，每个物理面板/极化子阵列独立发射 DFT 波束；
- 默认 `scheduler.max_mu_order: auto`，会根据 RF architecture 自动解析；
- 默认 4 TXRUs，因此默认最大同时发射模拟波束数 = 4，默认最大 MU order = 4；
- 支持 1 站点、3 站点等边三角形、7 站点六边形站群；
- 支持 `per_site_joint` 站点域调度：同一站点的 3 个扇区一起调度，UE 只上报本服务站点 3 个扇区内的候选波束；
- `exhaustive` 穷举调度新增站点域拆分、panel 约束剪枝、零上界剪枝和 branch-and-bound 上界剪枝；
- 新增仿真进度输出；
- 新增 `docs/yaml_parameter_reference.md`，自包含说明 YAML 参数含义、取值范围和注意事项。

> 说明：默认配置要求使用真实 Sionna TR 38.901 UMa/UMi/RMa 信道和 Sionna SYS PDSCH BLER/ILLA 链路抽象；如果 Sionna 后端不可用会直接报错，避免静默回退。调试 fallback 可改用 `scenario.channel_model: numpy_geometric_uma` 和 `link_abstraction.mode: fallback_precomputed_table`。

---

## 1. 快速运行

进入项目根目录：

```bash
cd sionna_sls_beam_mgmt_v2_4
```

检查环境：

```bash
/home/zhangwei/anaconda3/envs/tf_sionna_rt/bin/python scripts/check_env.py
```

服务器推荐命令，下行 PDSCH、目标 BLER 10%、greedy 调度、50 drops、每 drop 50 TTI：

```bash
cd /path/to/sionna_sls_beam_mgmt_v2_4
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
/home/zhangwei/anaconda3/envs/tf_sionna_rt/bin/python -m beam_sls.run \
  --config configs/v2_one_site_three_sector.yaml \
  --out runs/v2_4_pdsch_greedy_drop50_tti50 \
  --num-drops 50 \
  --num-tti 50 \
  --algorithm greedy
```

如果服务器没有 GPU 或希望由 TensorFlow 自动选择设备，可以去掉 `CUDA_VISIBLE_DEVICES=0`。

当前默认链路抽象为下行 PDSCH：

```yaml
system:
  target_bler: 0.1

link_abstraction:
  mode: sionna_sys_precomputed_bler
  mcs_table_index: 1
  mcs_category: 1   # Sionna SYS category 1 = PDSCH
```

运行默认配置：

```bash
/home/zhangwei/anaconda3/envs/tf_sionna_rt/bin/python -m beam_sls.run \
  --config configs/v2_one_site_three_sector.yaml \
  --out runs/v2_4_one_site_three_sector
```

快速调试运行，跳过覆盖热力图：

```bash
/home/zhangwei/anaconda3/envs/tf_sionna_rt/bin/python -m beam_sls.run \
  --config configs/v2_one_site_three_sector.yaml \
  --out runs/v2_4_smoke \
  --num-drops 1 \
  --num-tti 1 \
  --algorithm greedy \
  --skip-heatmap
```

关闭进度输出：

```bash
/home/zhangwei/anaconda3/envs/tf_sionna_rt/bin/python -m beam_sls.run \
  --config configs/v2_one_site_three_sector.yaml \
  --out runs/v2_4_quiet \
  --quiet
```

三站点等边三角形站点域调度：

```bash
/home/zhangwei/anaconda3/envs/tf_sionna_rt/bin/python -m beam_sls.run \
  --config configs/v2_three_site_triangle.yaml \
  --out runs/v2_4_three_site_greedy \
  --num-drops 5 \
  --num-tti 5 \
  --algorithm greedy \
  --skip-heatmap
```

七站点六边形站群：

```bash
/home/zhangwei/anaconda3/envs/tf_sionna_rt/bin/python -m beam_sls.run \
  --config configs/v2_seven_site_hex.yaml \
  --out runs/v2_4_seven_site_greedy \
  --num-drops 5 \
  --num-tti 5 \
  --algorithm greedy \
  --skip-heatmap
```

也可以直接在命令行覆盖 topology 和调度域：

```bash
/home/zhangwei/anaconda3/envs/tf_sionna_rt/bin/python -m beam_sls.run \
  --config configs/v2_one_site_three_sector.yaml \
  --layout three_site_triangle \
  --num-sites 3 \
  --domain-mode per_site_joint \
  --out runs/v2_4_three_site_override \
  --num-drops 1 \
  --num-tti 1 \
  --skip-heatmap
```

---

## 2. Topology：1/3/7 站点

默认配置为 1 个站点、3 个 sector/cell：

```yaml
topology:
  layout: one_site_three_sector
  num_sites: 1
  sectors_per_site: 3
  sector_azimuths_deg: [30.0, 150.0, 270.0]
  sector_width_deg: 120.0
  isd_m: 500.0
  bs_height_m: 25.0
```

运行后会输出：

```text
figures/topology.png
```

图中会标出 site、sector boresight、sector 边界、UE drop 和 ISD 标尺。

新增多站点布局：

```yaml
topology:
  layout: three_site_triangle
  num_sites: 3
  sectors_per_site: 3
  isd_m: 500.0
```

`three_site_triangle` 生成 3 个站点，任意两站点距离均为 `isd_m`，总小区数为 `3 * sectors_per_site`。

```yaml
topology:
  layout: seven_site_hex
  num_sites: 7
  sectors_per_site: 3
  isd_m: 500.0
```

`seven_site_hex` 生成 1 个中心站点和 6 个第一圈邻站，总小区数为 `7 * sectors_per_site`。当前实现是有限 7 站点站群，不做 wrap-around 边界复制。

UE drop 仍按 sector 扇形区域生成：

```yaml
ue_drop:
  num_ut_per_sector: 10
  distribution: uniform_in_sector
```

因此 7 站点、每站 3 扇区、每扇区 10 个 UE 时：

$$
\mathrm{num\_sites} = 7
$$

$$
\mathrm{num\_cells} = 21
$$

$$
\mathrm{num\_ues} = 210
$$

---

## 3. TRP 阵列和 DFT 码本

默认 TX 阵列：

```yaml
tx_array:
  model: tr38901_panel
  num_txru: 4
  num_ae: 1024
  M: 16
  N: 16
  P: 2
  Mg: 2
  Ng: 1
  Mp: 1
  Np: 1
  dH: 0.5
  dV: 0.5
```

AE 数校验：

$$
\begin{aligned}
\mathrm{num\_ae}
&= M \times N \times P \times M_g \times N_g \times M_p \times N_p \\
&= 16 \times 16 \times 2 \times 2 \times 1 \times 1 \times 1 \\
&= 1024
\end{aligned}
$$

DFT 空间码本不乘极化数 `P`：

$$
\begin{aligned}
H &= N \times N_g \times N_p = 16 \\
V &= M \times M_g \times M_p = 32 \\
\mathrm{full\ spatial\ codebook\ size} &= H \times V = 512 \\
\mathrm{beam\ vector\ length} &= H \times V \times P = 1024\ \mathrm{AEs}
\end{aligned}
$$

默认 SLS 扫描不是完整 512 个方向，而是均匀采样：

```yaml
tx_array:
  num_beams_h: 4
  num_beams_v: 4
  max_beams: 16
```

即每个活动码本扫描：

$$
\mathrm{num\_beams}_h \times \mathrm{num\_beams}_v = 16\ \mathrm{beams}
$$

---

## 4. RF architecture 与 MU order

v2.4 新增：

```yaml
rf_architecture:
  txru_connectivity: panel_polarization_subarray
  allow_independent_polarization_beams: true
  num_txru: 4
  max_parallel_beams_per_trp: auto

scheduler:
  max_mu_order: auto
  cap_mu_order_by_rf: true
```

### 4.1 默认情况 1：sub-connected / panel-polarization connected

配置：

```yaml
rf_architecture:
  txru_connectivity: panel_polarization_subarray
  allow_independent_polarization_beams: true
  num_txru: 4
```

含义：

```text
TXRU 0 -> panel 0, polarization 0
TXRU 1 -> panel 0, polarization 1
TXRU 2 -> panel 1, polarization 0
TXRU 3 -> panel 1, polarization 1
```

每个 TXRU 形成一个局部 panel-polarization DFT beam。两个极化允许使用不同 beam，因此：

$$
\mathrm{max\_parallel\_beams\_per\_trp} = \mathrm{num\_txru} = 4
$$

$$
\mathrm{scheduler.max\_mu\_order(auto)} = 4
$$

在默认 1-site 3-sector、每 sector 1 个 TRP 的情况下：

$$
\mathrm{per\ sector}: 4\ \mathrm{TX\ units} \times 16\ \mathrm{beams} = 64\ \mathrm{TX\ beam\ IDs}
$$

$$
\mathrm{network}: 3\ \mathrm{sectors} \times 64 = 192\ \mathrm{TX\ beam\ IDs}
$$

### 4.2 情况 1 但同一面板两个极化共享 beam

配置：

```yaml
rf_architecture:
  txru_connectivity: panel_polarization_subarray
  allow_independent_polarization_beams: false
```

含义：同一 panel 的两个极化共享一个空间 beam，不允许两个极化独立扫不同方向。默认 TRP 有 2 个物理面板，因此：

$$
\mathrm{max\_parallel\_beams\_per\_trp} = \mathrm{number\_of\_physical\_panels} = 2
$$

$$
\mathrm{scheduler.max\_mu\_order(auto)} = 2
$$

### 4.3 情况 2：fully-connected hybrid beamforming

配置：

```yaml
rf_architecture:
  txru_connectivity: fully_connected
  num_txru: 4
```

含义：每个 TXRU 都连接到整个 TRP 的 1024 AEs，每个 TXRU 可形成一个 full-array DFT beam。因此：

$$
\mathrm{max\_parallel\_beams\_per\_trp} = \mathrm{num\_txru} = 4
$$

$$
\mathrm{scheduler.max\_mu\_order(auto)} = 4
$$

这个模式下，每个同时发射的 beam 都是 full-array beam；这隐含了 fully-connected 或足够灵活的 hybrid RF 连接结构。

---

## 5. 固定垂直波束 / 电下倾角选择

TX 支持固定垂直 DFT beam：

```yaml
tx_array:
  vertical_beam_mode: fixed
  fixed_v_index: 3
```

也支持用覆盖仿真选择电下倾角：

```yaml
coverage_heatmap:
  fixed_vertical_beam_cdf:
    enabled: true
    candidate_v_indices: all
    horizontal_num_beams: 4
    selection_metric: mean_dbm
```

对于每个候选垂直 DFT index，程序固定该垂直 beam，只扫描水平 beam。每个覆盖点上，对所有水平扫描 beam 的 RSRP 做平均，然后统计覆盖 RSRP CDF。输出包括：

```text
figures/fixed_vertical_beam_cdf.png
metrics/fixed_vertical_beam_summary.csv
metrics/fixed_vertical_beam_samples.csv
fixed_vertical_beam_selection.json
```

---

## 6. 调度器

默认调度器为 greedy：

```yaml
scheduler:
  domain_mode: per_site_joint
  algorithm: greedy
  objective: sum_rate
  max_mu_order: auto
  use_panel_constraint: true
  exhaustive_pruning:
    enabled: true
    sort_by_upper_bound: true
    zero_upper_bound: true
    branch_and_bound: true
```

### 6.1 调度域

`scheduler.domain_mode` 支持：

```yaml
scheduler:
  domain_mode: per_site_joint
```

`per_site_joint` 表示站点域调度：

- 每个 UE 只在自己 `site_id` 对应站点的 3 个扇区内选择候选服务 beam；
- `topk_conflict_id` 和 `threshold_conflict_set` 的冲突 beam 也只来自该站点域；
- 调度器按 `site_id` 分组，每个站点独立运行一次 `greedy` 或 `exhaustive`；
- 各站点的调度结果会合并成一个全网 schedule；
- 链路层实际传输时仍使用合并后的全网同时发射结果计算真实 effective SINR、BLER 和 ACK，因此其他站点的已调度 beam 会作为实际干扰出现。

也可以使用全网调度：

```yaml
scheduler:
  domain_mode: global
```

`global` 下 UE 可以从全网所有 beam 中上报候选 beam，调度器也把所有 UE 放在一个全局问题里求解。旧配置名 `single_site_three_sector_independent` 仍兼容，内部等价为 `per_site_joint`。

### 6.2 Greedy 与穷举

`greedy` 推荐用于常规仿真。它每一步加入一个能带来最大目标增益的 UE/beam，直到达到 `max_mu_order` 或没有正增益候选。

`exhaustive` 会在给定上报候选集合内搜索最优组合。未剪枝时复杂度近似为：

$$
\sum_{q=1}^{Q} \binom{U}{q} K^q
$$

其中：

- `U` 是调度域内 UE 数；
- `K` 是每个 UE 上报的候选服务 beam 数；
- `Q` 是 `max_mu_order`。

在 `per_site_joint` 下，上式里的 `U` 是单个站点域内的 UE 数，而不是全网 UE 数。七站点时相当于做 7 个较小的站点内穷举，再合并结果。

### 6.3 穷举剪枝

当前穷举剪枝默认开启：

```yaml
scheduler:
  exhaustive_pruning:
    enabled: true
    sort_by_upper_bound: true
    zero_upper_bound: true
    branch_and_bound: true
```

剪枝方式：

- 站点域拆分：`per_site_joint` 下每个站点单独穷举，避免把多个站点的 UE 放进一个组合爆炸的全局搜索。
- 候选集预限制：穷举只搜索 UE 已上报的服务 beam。候选数由 `feedback.service_beam_top_k1` 和 `feedback.oracle_service_beam_top_k` 控制。
- panel 约束剪枝：若 `use_panel_constraint: true`，同一个 `(cell, trp, panel)` 同时只能选择一个 beam；违反该约束的分支直接跳过，不进入链路目标计算。
- 零上界剪枝：如果某个 UE 所有候选 beam 的单用户加权速率上界为 0，它不可能提高目标函数，会被跳过。
- branch-and-bound 上界剪枝：对每个 UE 计算“单用户、无干扰、无冲突惩罚”的最大加权速率，作为该 UE 在任何 MU 组合中的收益上界。搜索过程中，如果“当前已选组合目标 + 剩余 UE 最大可能上界”仍不超过当前最优值，则整棵分支跳过。
- 上界排序：先搜索单用户上界更高的 UE/beam，更快得到较好的当前最优值，从而让 branch-and-bound 更早生效。

上述剪枝不会改变 `exhaustive` 在当前上报候选集合内的最优性。它只跳过违反硬约束的组合，或跳过理论上不可能超过当前最优解的分支。

### 6.4 MU order

`max_mu_order: auto` 时，程序根据 RF architecture 自动设置最大同时调度 UE 数。若手动写整数，例如：

```yaml
scheduler:
  max_mu_order: 3
  cap_mu_order_by_rf: true
```

实际 MU order 会被 RF 物理并发 beam 数截断：

$$
\mathrm{effective\_max\_mu\_order}
= \min(3,\ \mathrm{max\_parallel\_beams\_per\_trp})
$$

在 `per_site_joint` 下，这个 MU order 是每个站点域的 MU order；全网同一 TTI 最多可能调度：

$$
\mathrm{num\_sites} \times \mathrm{effective\_max\_mu\_order}
$$

---

## 7. 进度输出

默认开启：

```yaml
progress:
  enabled: true
```

运行时会看到类似：

```text
[init] RF=panel_polarization_subarray, tx_units/TRP=4, max_mu_order=4
[run] drops=10, tti/drop=50, schemes=full_gamma,baseline,..., beams=192, tx_units/sector=4
[drop 1/10] topology + channel generation
[drop 1/10] channel backend=fallback_numpy_for_sionna_tr38901_uma; computing Gamma measurement
[drop 1/10] scheduling 4 feedback schemes
[drop 1/10] finished
[coverage] generating coverage heatmap and fixed-vertical-beam CDF
[done] outputs written to ...
```

---

## 8. 输出文件

主要输出：

```text
resolved_config.yaml
array_config_summary.json
rf_architecture_summary.json
sionna_import_probe.json
link_abstraction_status.json
figures/topology.png
figures/coverage_heatmap.png
figures/best_beam_heatmap.png
figures/fixed_vertical_beam_cdf.png
metrics/summary.csv
metrics/link_tti.csv
metrics/schedules.csv
metrics/scheduler_stats.csv
metrics/beams.csv
metrics/reports.csv
metrics/ues.csv
metrics/sites.csv
metrics/sectors.csv
```

`rf_architecture_summary.json` 会记录最终解析出的：

```text
txru_connectivity
allow_independent_polarization_beams
tx_units_per_trp
max_parallel_beams_per_trp
effective_beam_scope
each TX unit's panel/polarization mapping
```

`metrics/drops.csv` 记录每个 drop 的网络规模和后端：

```text
num_sites
num_cells
num_ues
num_beams
scheduler_domain_mode
channel_backend
link_adaptation_backend
```

`metrics/reports.csv` 中的 `report_json` 会记录 UE 上报内容。站点域调度时，每条 report 会包含：

```text
ue_id
site_id
serving_cell
candidates
```

候选 beam 的 `beam_id` 形如：

```text
c<cell>t<site/trp>p<panel>b<beam>
```

站点域下，UE 的候选服务 beam 应只来自与 `site_id` 相同的 `t<site/trp>`。

`metrics/scheduler_stats.csv` 用于解读调度复杂度和剪枝效果。常用字段：

```text
drop
scheme
domain_mode
domain_id
algorithm
num_reports_input
num_reports_with_candidates
num_reports_after_pruning
max_mu_order
raw_assignment_count
assignment_count_after_zero_prune
evaluated_assignment_count
panel_pruned_count
bound_pruned_count
zero_upper_bound_pruned_reports
best_objective_value
num_scheduled
```

字段含义：

- `domain_id`：站点域调度时为 `site_id`；`all` 行是所有站点的合计。
- `raw_assignment_count`：穷举在剪枝前、基于上报候选集合需要考虑的组合数。
- `assignment_count_after_zero_prune`：移除零上界 UE 后剩余的理论组合数。
- `evaluated_assignment_count`：真正进入目标函数计算的组合数。
- `panel_pruned_count`：因同一 panel/TX unit 重复用 beam 而跳过的组合数。
- `bound_pruned_count`：被 branch-and-bound 上界剪掉的分支数。
- `zero_upper_bound_pruned_reports`：因单用户速率上界为 0 而移除的 UE report 数。

一般可用下面的比例粗略看剪枝收益：

$$
\frac{\mathrm{evaluated\_assignment\_count}}{\mathrm{raw\_assignment\_count}}
$$

比例越小，说明穷举实际评估的组合越少。该比例只反映调度搜索复杂度，不代表链路性能。

---

## 9. YAML 参数说明

完整 YAML 参数说明见：

```text
docs/yaml_parameter_reference.md
```

该文档逐项说明参数含义、建议取值范围、默认值逻辑和注意事项，不需要依赖任何历史版本说明。


## v2.4.1 Sionna TensorFlow-adapter hotfix

This hotfix keeps the v2.4 simulator design but fixes the Sionna 1.0.2 adapter for environments where Sionna SYS/TR38901 are TensorFlow-backed. In the original v2.4 adapter, `torch` tensors were passed to `sionna.sys.InnerLoopLinkAdaptation` and `sionna.sys.PHYAbstraction`, which can fail with:

```text
TypeError: Cannot convert the argument `type_value`: torch.int32 to a TensorFlow DType.
```

The hotfix uses TensorFlow tensors for Sionna SYS and TR38901 calls. PyTorch can still be installed and detected, but the actual Sionna 1.0.2 link-adaptation/channel calls in this package use TensorFlow tensors.

The default config now sets:

```yaml
sionna:
  fallback_to_numpy_if_unavailable: false
  tensor_backend: tensorflow
```

This means that if the requested Sionna TR38901/SYS backend cannot be initialized, the run stops with the real error instead of silently using the NumPy fallback. For fallback/debug runs, set `scenario.channel_model: numpy_geometric_uma` and `link_abstraction.mode: fallback_precomputed_table`.

## v2.4.2 Sionna TR38901 adapter hotfix

This hotfix fixes a Sionna CIR axis-order bug in the strict TR38901 backend. In v2.4.1, the adapter converted the Sionna CIR tensor with the wrong `einsum` index order and could fail with a broadcast error such as:

```text
ValueError: operands could not be broadcast together with remapped shapes ...
```

For Sionna 1.0.2 in the tested environment, CIR coefficients are interpreted as:

```text
a   [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, time]
tau [batch, num_rx, num_tx, num_paths]
```

and are converted to the simulator internal tensor:

```text
H [num_ue, num_tx_unit, num_freq, num_rx_ant, num_tx_ant]
```

The hotfix also reconciles Sionna `PanelArray` antenna dimensions with the simulator's explicit 3GPP `P` polarization dimension. If Sionna returns a spatial antenna dimension while the simulator beam vectors explicitly contain polarization blocks, the channel is expanded over the polarization blocks so the DFT beam vectors and channel tensor dimensions match.

Use the strict configuration to require the real Sionna TR38901 backend:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 \
/home/zhangwei/anaconda3/envs/tf_sionna_rt/bin/python -m beam_sls.run \
  --config configs/v2_one_site_three_sector_sionna_strict.yaml \
  --out runs/v2_4_2_sionna_strict_check \
  --num-drops 1 \
  --num-tti 1 \
  --algorithm greedy
```

With `sionna.fallback_to_numpy_if_unavailable: false`, any remaining Sionna TR38901 API or topology error is raised immediately instead of silently falling back.
