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

> 说明：默认配置要求使用真实 Sionna TR 38.901 UMa/UMi/RMa 信道和 Sionna SYS PDSCH BLER/ILLA 链路抽象；如果 Sionna SYS 后端不可用会直接报错。链路自适应不再提供本地 MCS/BLER fallback；仅信道调试仍可显式使用 `scenario.channel_model: numpy_geometric_uma`。

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
  --olla-warmup-tti 100 \
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
  # Extra TTIs per drop; OLLA updates normally, but these rows are not counted.
  olla_warmup_tti: 100
```

在旧的非连续模式下，`system.num_tti_per_drop` 只表示正式统计 TTI 数。每个
drop 实际运行 `olla_warmup_tti + num_tti_per_drop` 个 TTI；warmup 正常抽样
ACK 并更新 OLLA，但不写入 `link_tti.csv`，也不进入吞吐、BLER 和 CDF。连续模式
的 warmup 同样不计入统计，但会逐 TTI 推进信道、重新 PF 调度并更新 PF/OLLA。

连续 TTI 可配置为：

```yaml
system:
  continuous_tti:
    enabled: true
    duration_ms: 2.0
    # 非 null 时直接指定正式统计 TTI 总数，并优先于 duration_ms。
    num_tti: null
    # null 时沿用 link_abstraction.olla_warmup_tti。
    warmup_tti: null

ue_drop:
  speed_kmh: 3.0
```

启用后，每个 drop 固定 UE 位置、路损和阴影衰落，只生成一次基准信道。
每 TTI 用 UE 速度对应的最大多普勒
`f_D = v f_c / c` 和 Jakes 相关系数
`rho = J0(2*pi*f_D*T_TTI)` 推进小尺度信道。每个 drop 只进行一次测量和
反馈，但每个 TTI 都使用当前 PF 状态重新选择 UE/波束，并在真实链路评估后更新
PF/OLLA。`num_tti` 可直接指定正式 TTI 总数；为 `null` 时才由
`duration_ms / slot_duration_ms` 换算。

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

三站点、每站点 3 TRP、全网联合调度且最多 36 用户并发：

```bash
/home/zhangwei/anaconda3/envs/tf_sionna_rt/bin/python -m beam_sls.run \
  --config configs/v2_three_site_global_36ue.yaml \
  --out runs/v2_4_three_site_global_36ue
```

该配置的统计口径、候选数选择和 36 并发验证过程见 `修改说明v2.4.md`。

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

默认共享码本按一个物理面板定义，DFT 空间码本不乘极化数 `P`：

$$
\begin{aligned}
H_{\mathrm{panel}} &= N = 16 \\
V_{\mathrm{panel}} &= M = 16 \\
\mathrm{per\ panel\ spatial\ codebook\ size} &= H_{\mathrm{panel}} \times V_{\mathrm{panel}} = 256 \\
\mathrm{compact\ beam/channel\ TX\ dimension} &= M \times N \times P = 512
\end{aligned}
$$

默认 SLS 固定 `measurement.tx_panel_index: 0`，从 256 个方向中均匀采样：

```yaml
tx_array:
  num_beams_h: 4
  num_beams_v: 4
  max_beams: 16

measurement:
  tx_panel_index: 0
  use_panel_channel_views: true
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
  allow_independent_polarization_beams: false
  num_txru: 4
  max_parallel_beams_per_trp: auto

scheduler:
  max_mu_order: auto
  cap_mu_order_by_rf: true
```

### 4.1 默认情况：双极化共享波束、动态 TXRU 分配

配置：

```yaml
rf_architecture:
  txru_connectivity: panel_polarization_subarray
  allow_independent_polarization_beams: false
  num_txru: 4

measurement:
  tx_panel_index: 0
  use_panel_channel_views: true
```

同一 panel 的两个极化共享空间波束。码本不绑定具体 TXRU，调度器从 TRP
共享码本中任意选择波束。默认 TRP 有 2 个物理面板，因此：

$$
\mathrm{max\_parallel\_beams\_per\_trp} = \mathrm{number\_of\_physical\_panels} = 2
$$

完整 TRP 信道始终以 1024 维保存。SLS 只从完整张量中提取
`tx_panel_index` 对应的 `M*N*P=512` 维计算视图；实际传输则根据调度器为
每个波束动态分配的物理面板提取对应视图。`use_panel_channel_views: false`
仅用于改回 1024 维零填充码本做等价性对照，不会改变完整信道的保存方式。

### 4.2 兼容模式：极化独立波束

设置 `allow_independent_polarization_beams: true` 后，恢复每个
panel-polarization/TXRU 子阵列独立码本和独立波束。

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

调度器支持普通 greedy、硬冲突 greedy、自适应 lambda greedy 和小规模 exhaustive：

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

硬冲突 greedy：

```yaml
scheduler:
  algorithm: hard_conflict_greedy
```

它把每个 `(UE, beam)` 当作一个节点，按 SU rate 从高到低选择。选中节点后，只删除该 UE 的其他候选节点、与该节点存在任一方向冲突的候选节点，以及违反 TX unit 约束的候选节点。某个 UE 的一个 beam 冲突不会导致该 UE 的其他 beam 被删除。多个候选的调度指标完全相同时，有限反馈 greedy 和硬冲突 greedy 都优先选择与当前仍可选的其他 UE 候选冲突最少的节点；若冲突影响仍相同，再按原候选顺序或 `(UE ID, beam index)` 确定性打破平局。

自适应 lambda：

```yaml
scheduler:
  algorithm: adaptive_lambda_greedy
  adaptive_lambda_alpha: 0.2
```

每个调度域内使用 `lambda = alpha * median(candidate SU rate [Mbps])`。也可以保留 `algorithm: greedy`，同时设置 `conflict_penalty_mode: adaptive`。实际使用的 lambda、中位 SU rate 和候选样本数会写入 `metrics/scheduler_stats.csv`。

### 6.1 测量域与调度域

v2.12 现行设计统一为“测量域 = 调度域 = UE 所属静态调度簇”。
`measurement.domain_mode` 已废弃。UE 先按宽带平均 RSRP 最大原则关联 cell，
再继承该 cell 唯一的 `scheduling_cluster`。服务候选只来自 serving cell，
干扰测量覆盖所属簇的全部 cell/beam。

```yaml
scheduler:
  cluster_mode: per_site  # per_cell | per_site | global | custom
```

静态簇必须完整覆盖全网 cell 且互不交叠。各簇调度结果最后合并；真实链路评估
仍使用全网同时发射波束计算 effective SINR、BLER 和 ACK。由于 UE 不测量簇外
波束，调度阶段等价于将簇外激活假设设为 0，簇边缘预测可能偏乐观。

为了保留全局 `beam_index` 编号，稀疏 Gamma 仍支持 `gamma[u, m, n]`：
`m` 属于 serving-cell 服务波束集合，`n` 属于测量域干扰波束集合。未测量的
调度域内干扰按 `scheduler.unknown_interference_policy: zero` 处理；真实链路
评估仍会计算它产生的实际干扰。

这里的裁剪只作用于调度前的测量、上报和预测信息，不表示实际传输时忽略域外干扰。调度器先在自己的调度域内选择 UE/beam；各调度域的结果随后会合并到同一个 TTI。链路层评估 ACK/BLER 时，不再查询一个预先算好的全网 Gamma 表，而是用真实信道 `H`、合并后的 `schedule.links`、已选 TX/RX beam 重新计算本 TTI 的 effective SINR。

因此需要区分两类计算：

- 调度前候选 Gamma：如果全网有 `B` 个 beam，完整预计算复杂度近似随下面的量增长：

$$
U \times B^2
$$

集合解耦后变成：

$$
U \times B_{\text{service}} \times B_{\text{measurement}}
$$

这里节省的是候选 beam pair 的测量/预测开销。

- 调度后真实干扰：只对本 TTI 实际发射的 link 集合计算，复杂度近似随下面的量增长：

$$
L^2
$$

其中 `L` 是当前 TTI 已调度 link 数，通常远小于全网候选 beam 数。

所以，真实传输仍然考虑全网已调度 beam 的干扰；节省的是调度前不必要的全网候选 Gamma 预计算。可以理解为：

$$
\text{全网真实干扰必须算；全网候选 Gamma 不必算。}
$$

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
在 `single_site_three_sector_independent` 下，`U` 是单个 sector 内的 UE 数。

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

- 调度域拆分：`per_site_joint` 下每个站点单独穷举，`single_site_three_sector_independent` 下每个 sector 单独穷举，避免把多个独立域的 UE 放进一个组合爆炸的全局搜索。
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
[run] drops=10, warmup_tti/drop=100, measured_tti/drop=50, schemes=full_gamma,baseline,..., beams=192, tx_units/sector=4
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
metrics/cell0_local_nack_rate.csv
metrics/cell0_local_nack_rate_status.csv
metrics/schedules.csv
metrics/scheduler_stats.csv
metrics/scheduler_iterations.csv
metrics/scheduled_ue_su_throughput.csv
metrics/ue_goodput.csv
metrics/system_tti_goodput.csv
metrics/system_drop_avg_goodput.csv
metrics/schedule_similarity.csv
metrics/schedule_similarity_by_drop.csv
metrics/su_snr_samples.csv
metrics/su_snr_max_per_ue.csv
metrics/su_snr_summary.csv
metrics/beams.csv
metrics/reports.csv
metrics/topk_interference_details.csv
metrics/ues.csv
metrics/sites.csv
metrics/sectors.csv
figures/ue_goodput_cdf.png
figures/system_tti_goodput_cdf.png
figures/system_drop_avg_goodput_cdf.png
figures/reported_su_snr_cdf.png
figures/reported_max_su_snr_per_ue_cdf.png
figures/cell0_local_nack_rate/<scheme>/drop_<drop>.png
```

`baseline_no_interference_upper_bound` 是 baseline 原调度集合在“波束间干扰强制为零”条件下重新运行链路层得到的诊断上界，并出现在 `link_tti.csv`、`summary.csv` 和吞吐 CDF 中。`ue_goodput.csv` 对每个 `(drop, UE)` 跨全部正式 TTI 求平均，未调度 UE 按零吞吐计入。`system_tti_goodput.csv` 显式保留零吞吐 TTI；`system_drop_avg_goodput.csv` 对每个 drop 的完整正式统计窗口求平均，因此这些 CDF 不会产生幸存者偏差。

`metrics/link_tti.csv` 按“已调度链路 × 正式 TTI”记录中间结果：
`predicted_sinr_db`/`predicted_mcs` 是调度器预测值，
`effective_sinr_db`/`actual_mcs` 是加入实际共调度干扰并执行链路自适应后的值。
`metrics/scheduled_ue_su_throughput.csv` 另按已调度波束记录调度前的
`su_snr_db`/`su_mcs`。`metrics/scheduler_iterations.csv` 的
`selected_tiebreak_snr_db`/`selected_tiebreak_mcs` 记录 greedy 每轮最终选中候选
用于平局判定的预测值。

`system.tx_power_dbm` 表示每个 TRP 的总发射功率。代码先转换到线性功率，
再均分给该 TRP 的所有物理面板；不会除以全网小区数、站点数或 TRP 数。
`panel_power_mode` 是已弃用兼容字段，不再改变该语义。解析后的每 TRP/每面板功率
会写入 `resolved_config.yaml`、`array_config_summary.json` 和 `metrics/drops.csv`。

`summary.csv/json` 对每个方案新增 `tbler_zero_ratio`、
`avg_scheduled_users_per_tti`、`p05_effective_sinr_db`、
`p50_effective_sinr_db` 和 `p95_effective_sinr_db`。这些指标只使用 warm-up
之后的正式 TTI；平均调度用户数的分母包含零调度 TTI。

小区 0 的局部 NACK 统计以每个 drop 中实际关联到小区 0 的 UE 为对象，只沿该 UE
被调度的正式 TTI 样本前进，每满 500 个样本计算一个 NACK rate 曲线点。
不足 500 的尾段不进入曲线，数量记录在 `cell0_local_nack_rate_status.csv`；即使某个
drop 没有完整窗口，也会生成说明性图片。

已有 run 可以用独立脚本重新选择曲线和样式，无需重跑仿真：

```bash
python scripts/plot_cdf.py runs/YOUR_RUN \
  --metric ue_goodput \
  --schemes full_gamma baseline baseline_no_interference_upper_bound \
  --font-size 13 \
  --output runs/YOUR_RUN/figures/ue_goodput_compare.pdf
```

完整的字体、标题、坐标范围、图例、颜色、线型、线宽、marker 和 ECDF
数据导出配置见 `configs/cdf_plot_example.yaml`：

```bash
python scripts/plot_cdf.py runs/YOUR_RUN \
  --config configs/cdf_plot_example.yaml
```

支持的 `--metric` 包括 `link_goodput`、`ue_goodput`、`effective_sinr`、
`tbler`、`olla_offset`、`reported_su_snr` 和 `reported_max_su_snr`。
其中 `baseline_no_interference_upper_bound` 即 baseline 调度在强制无波束间
干扰条件下得到的 SU 参考曲线。

调度相似度使用完全相同的 `(UE, beam_index)` 二元组。主指标为 Jaccard 相似度，同时输出按较大调度集合归一化的相同比例、overlap coefficient 和完全一致比例。

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
measurement_domain_mode
scheduling_cluster_mode
num_scheduling_clusters
channel_backend
link_adaptation_backend
```

`metrics/measurement_domains.csv` 逐 `(drop, UE)` 记录 `serving_cell`、
`site_id`、`scheduling_cluster`、`cluster_mode`、`serving_average_rsrp_w`、
`service_cell_ids`、`measured_cell_ids`、服务候选 beam 数和簇内测量
cell/beam 数，以及去重后的 `num_reported_beams`。应检查每个 UE 的
`measured_cell_ids` 恰好等于其静态簇 cell 集合。

`metrics/reports.csv` 中的 `report_json` 会记录 UE 上报内容：

```text
ue_id
site_id
serving_cell
scheduling_cluster
measured_interference_beams
candidates
```

`topk_conflict_id` 的每个服务候选还包含 `interference_details`。每个上报的
强干扰 beam 记录 `service_snr_db`（仅有噪声时的 \(S/N\)）、
`pair_sinr_db`（该干扰 beam 存在时的 \(S/(I+N)\)）和
`sinr_loss_db = service_snr_db - pair_sinr_db`，以及相同链路抽象后端得到的
`service_mcs`、`pair_mcs` 和
`mcs_loss = service_mcs - pair_mcs`。`service_outage`、`pair_outage`
用于区分最低 MCS 与已经低于目标 BLER 的情况。相同内容同时按每个
`(drop, UE, service beam, interferer beam)` 一行写入
`metrics/topk_interference_details.csv`，并保留 site/cell/cluster、服务候选排名
和干扰 beam 排名。该文件基于每个 drop 唯一一次的 Gamma 测量，因此连续 TTI
模式下不会为每个 TTI 重复写相同行。

候选 beam 的 `beam_id` 形如：

```text
c<cell>t<site/trp>p<panel>b<beam>
```

UE 的候选服务 beam 应只来自 `serving_cell`；测量干扰 beam 应全部属于
`scheduling_cluster`。

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

This means that if the requested Sionna SYS backend cannot be initialized, the run stops with the real error. Link adaptation has no local MCS/BLER fallback. For channel-only debugging, `scenario.channel_model: numpy_geometric_uma` remains available.

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

The Sionna adapter requires the official `PanelArray` antenna dimension,
including polarization, to exactly match the configured AE count. It explicitly
permutes Sionna's panel/polarization/vertical-fast antenna order into the
simulator's polarization/global-row-major codebook order; dimensions are never
repeated, averaged, or silently truncated.

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
## v2.12 静态协同簇（现行设计）

本节替代此前“测量域与调度域独立配置”的说明。现行模型统一为：

```text
测量域 = 调度域 = UE 所属的静态调度簇
```

每个 cell 必须且只能属于一个调度簇；所有 cell 必须被完整覆盖。UE 先按宽带
平均 RSRP 最大原则关联 serving cell，再唯一归属于拥有该 cell 的调度簇。对
UE `u` 和 cell `c`：

$$
\overline{P}^{\mathrm{RSRP}}_{u,c}
=
\max_{b\in\mathcal B_c,\ q\in\mathcal Q_u}
\left[
P_{\mathrm{TX}}\frac{1}{|\mathcal F|}
\sum_{f\in\mathcal F}
\left|q^{H}H_{u,\mathrm{tx}(b)}(f)w_b\right|^2
\right]
$$

$$
c_u^\star=\arg\max_c\overline{P}^{\mathrm{RSRP}}_{u,c}.
$$

完全相同时用较小 `cell_id` 确定性打破平局。服务候选只来自 serving cell；
干扰测量覆盖所属簇的全部 cell/beam；调度器只联合处理本簇 UE 的报告。本簇
之外的波束不进入调度预测，但合并各簇 schedule 后的真实链路评估仍计算全网
实际干扰，因此簇边缘预测可能偏乐观。

```yaml
scheduler:
  cluster_mode: per_site  # per_cell | per_site | global | custom
```

自定义静态簇：

```yaml
scheduler:
  cluster_mode: custom
  static_clusters:
    - cluster_id: 0
      cell_ids: [0, 1, 2, 3, 4, 5]
    - cluster_id: 1
      cell_ids: [6, 7, 8, 9, 10, 11]
```

`custom` 配置存在重复 cell、未知 cell、空簇或漏配 cell 时会直接报错。
`measurement.domain_mode` 不再参与仿真。旧 `scheduler.domain_mode` 仅在
未配置 `cluster_mode` 时兼容映射为 `per_cell`、`per_site` 或 `global`。

四种静态划分的准确含义：

- `per_cell`：每个 cell 单独成簇；三扇区站点会形成三个独立簇。
- `per_site`：同一 `site_id` 下全部 cell 成一个簇；不同站点互不重叠。
- `global`：全网全部 cell 组成唯一一个簇。
- `custom`：严格使用 `scheduler.static_clusters` 给出的 cell 列表划分。

主要阶段完成后会输出 `elapsed=<seconds>s`，并写入
`metrics/runtime_phases.csv`。Drop 级 phase 包括
`topology_generation`、`channel_generation`、`average_rsrp_association`、
`static_cluster_preparation`、`gamma_measurement` 和
`feedback_generation`；case 级 phase 包括 `scheduler` 和
`link_evaluation`。
