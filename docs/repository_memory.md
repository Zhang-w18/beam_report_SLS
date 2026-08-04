# 仓库记忆：Sionna SLS Beam Management Platform

最后更新：2026-07-24

本文件是供后续代码分析和修改任务快速建立上下文的稳定摘要。先看这里，再按函数
和配置键定向读取源码。若本文与代码或测试冲突，以代码和测试为准，并在架构改动后
更新本文。

## 1. 项目目标与入口

本项目是波束域系统级仿真平台，用于研究多站点/多扇区拓扑、SLS 波束测量、
有限反馈、MU-MIMO 调度、链路自适应、OLLA、吞吐和公平性。

主要入口：

- CLI：`beam_sls/run.py::main`
- 仿真总入口：`beam_sls/sim.py::run_simulation`
- 默认配置：`beam_sls/config.py::DEFAULT_CONFIG`
- YAML 配置：`configs/*.yaml`
- 运行方式：

```bash
python -m beam_sls.run \
  --config configs/phase1_single_cell.yaml \
  --out runs/example \
  --num-drops 1 \
  --num-tti 2 \
  --skip-heatmap
```

## 2. 代码地图

| 模块 | 主要职责 | 优先关注的符号 |
|---|---|---|
| `beam_sls/run.py` | CLI 参数覆盖、启动仿真 | `main` |
| `beam_sls/config.py` | 默认配置、YAML 合并、部分配置解析 | `DEFAULT_CONFIG`, `load_config` |
| `beam_sls/topology.py` | site/sector/UE 数据结构、站点布局和 UE drop | `Topology`, `make_topology` |
| `beam_sls/codebook.py` | 3GPP 风格阵列参数、DFT 码本、BeamId、panel 维度映射 | `ArrayConfig`, `BeamId`, `build_network_tx_beams`, `extract_panel_tx_dimension` |
| `beam_sls/rf.py` | RF architecture、TX unit、动态波束能力和 MU order | `RFArchitecture`, `resolve_rf_architecture`, `resolved_max_mu_order` |
| `beam_sls/channel.py` | NumPy/Sionna 信道生成、Sionna CIR 转换、连续 TTI 多普勒推进 | `ChannelRealization`, `generate_channel`, `SionnaTR38901Adapter`, `DopplerChannelEvolver` |
| `beam_sls/measurement.py` | service power、Gamma、RX beam 选择和 CPU/GPU 测量 | `MeasurementResult`, `SparseGamma`, `compute_gamma_measurement` |
| `beam_sls/feedback.py` | 从测量结果构造不同反馈方案的 UE report | `ServiceCandidate`, `UEReport`, `make_reports` |
| `beam_sls/evaluation.py` | feedback × algorithm 实验矩阵和 case ID | `EvaluationCase`, `EvaluationPlan`, `resolve_evaluation_plan` |
| `beam_sls/scheduler.py` | 调度域、greedy/exhaustive/硬冲突算法、PF 目标 | `schedule`, `_evaluate_assignments`, `update_pf_throughput` |
| `beam_sls/link_adaptation.py` | Sionna SYS 链路抽象、调度查表加速、标准 TBS | `SionnaSYSAdapter`, `make_link_adapter`, `SchedulerLinkLookup` |
| `beam_sls/mcs.py` | TS 38.214 PDSCH Table 1 与独立标准 TBS/速率工具 | `MCS_TABLE`, `tbs_bits_from_mcs`, `rate_mbps_from_mcs` |
| `beam_sls/link.py` | 调度后真实 SINR、EESM、ACK/NACK、OLLA、单/多 TTI 链路评估 | `realized_sinr_grid`, `run_tti_loop`, `run_one_tti` |
| `beam_sls/sim.py` | 整合 drop/TTI/case 循环、状态生命周期、指标输出和绘图 | `run_simulation`, `summarize_results`, `make_plots` |
| `beam_sls/coverage.py` | coverage heatmap、固定垂直波束 CDF | `compute_coverage_heatmap_standard_sampling`, `compute_fixed_vertical_beam_cdf` |
| `beam_sls/plotting.py` | CDF、柱状图、heatmap、topology 图 | `plot_cdf`, `plot_topology` |
| `beam_sls/utils.py` | 单位换算、噪声、随机数、CSV/JSON 写出 | `occupied_bandwidth_hz`, `write_csv`, `write_json` |

## 3. 主数据流

```text
YAML / DEFAULT_CONFIG
        ↓
resolve RF + array + codebook
        ↓
for each drop:
  topology / UE positions
        ↓
  channel H + fixed large-scale state
        ↓
  Gamma measurement + selected RX beams
        ↓
  feedback reports
        ↓
  evaluation case / scheduler
        ↓
  merged schedule
        ↓
  true post-scheduling interference + EESM
        ↓
  MCS / TBLER / ACK-NACK / OLLA
        ↓
  per-TTI rows → per-UE / per-drop / aggregate metrics and plots
```

每个 drop 只执行一次“测量 → feedback → 调度”。连续 TTI 模式在后续 TTI
复用该测量、report 和 schedule，仅推进小尺度信道并执行真实链路评估、
HARQ 风格 ACK/NACK 和 OLLA 更新。

## 4. 核心数组和索引约定

### 4.1 信道

内部频域信道主要形状：

```text
H[UE, TX_UNIT, FREQ, RX_AE, TX_AE]
```

Sionna CIR 入口约定：

```text
a[BATCH, RX, RX_ANT, TX, TX_ANT, PATH, TIME]
tau[BATCH, RX, TX, PATH]
```

`sionna_cir_to_internal_frequency_response` 负责转换到内部顺序，并显式映射
Sionna PanelArray 与本项目码本的天线顺序。不要通过简单 reshape 或静默截断修复
天线轴不匹配。

在 compact panel channel 模式下，测量和实际链路计算可能使用
`extract_panel_tx_dimension` 取得选定物理 panel 的 TX 维度。

### 4.2 BeamId 和资源

`BeamId` 记录 cell、TRP、panel、beam、global index 和 TX unit。调度资源约束取决于
RF architecture：

- 兼容/固定 panel 模式通常按 `panel_key()` 限制；
- 动态波束分配模式按 `trp_key()` 计数，并使用
  `_resolved.max_parallel_beams_per_trp` 作为容量；此外，同一 TRP 的同一个本地
  beam/codeword 在一个 TTI 内最多分配给一个 UE。

不要假设“一个 beam 永久绑定一个 TXRU/panel”；当前共享码本模式支持调度后动态
分配物理 panel。

### 4.3 Measurement/Gamma

现行约定为“测量域 = 调度域 = UE 所属静态调度簇”。
`measurement.domain_mode` 已废弃。UE 通过最佳 TX/RX 波束对的频点平均接收功率
最大值关联 serving cell，并继承该 cell 唯一的 cluster。服务候选只来自
serving cell；Gamma 干扰维覆盖该 cluster 全部 beam。簇外波束不进入调度预测，
合并 schedule 后的真实链路评估仍计算全网实际干扰。

`MeasurementResult` 的核心字段：

```text
service_power_w[UE, SERVICE_BEAM]
gamma[UE, SERVICE_BEAM, INTERFERER_BEAM]
selected_rx_beam[UE, SERVICE_BEAM]
su_mcs[UE, BEAM]
su_snr_db[UE, BEAM]
```

Gamma 的方向必须保持为“某 UE、某服务 beam、某干扰 beam”。完整 Gamma 可能使用
`SparseGamma` 仅保存 UE 调度域内的 block。

## 5. Drop、TTI 和状态生命周期

### 5.1 Drop

- 每个 drop 重新生成 topology、UE 位置和信道大尺度状态。
- drop 是独立随机实现。
- OLLA 和 PF 状态不能从一个随机 drop 泄漏到下一个 drop。
- 不同 evaluation case 应保持各自独立的 PF/OLLA 状态。

### 5.2 旧的非连续 TTI 模式

- `system.num_tti_per_drop` 决定正式统计 TTI 数。
- 一个 drop 内通常复用本 drop 的信道和 schedule。
- `olla_warmup_tti` 期间正常抽样 ACK、更新 OLLA，但不写正式指标行。

### 5.3 连续 TTI 模式

配置：

```yaml
system:
  continuous_tti:
    enabled: true
    duration_ms: 2.0
    num_tti: null
    warmup_tti: null

pdsch:
  slot_duration_ms: 0.125

ue_drop:
  speed_kmh: 3.0
```

规则：

- `num_tti` 为正整数时直接决定正式统计 TTI 数，并优先于 `duration_ms`；
- `num_tti=null` 时，`duration_ms / slot_duration_ms` 必须是正整数；
- `warmup_tti` 可直接设置 warmup 数；为 `null` 时沿用 `olla_warmup_tti`；
- UE 位置、路损、阴影和基准空间/时延结构在 drop 内固定；
- `DopplerChannelEvolver` 用 UE 速度和载频计算最大多普勒：

```text
f_D = v * f_c / c
rho = J0(2*pi*f_D*T_TTI)
g[t+1] = rho*g[t] + sqrt(1-rho^2)*z[t]
```

- 当前实现使用每个 `(UE, TX unit)` 一个复高斯小尺度状态，避免按完整天线张量生成
  随机创新；它保留基准空间和时延特征，是低开销的 link-wise Jakes 近似。
- 每个 drop 只测量、生成反馈一次；每个 warmup 和正式 TTI 都使用当前 PF 状态
  重新调度，并在真实链路评估后更新 PF/OLLA。

## 6. RF architecture 与码本

主要模式：

- `panel_polarization_subarray`：sub-connected 类架构；
- `fully_connected`：全连接混合波束赋形。

需要联合理解：

- `tx_array` 的 M/N/P/Mg/Ng/Mp/Np；
- `rf_architecture.txru_connectivity`；
- `allow_independent_polarization_beams`；
- `max_parallel_beams_per_trp`；
- `scheduler.max_mu_order`；
- `measurement.tx_panel_index` 和 compact panel view。

`system.tx_power_dbm` 固定表示每个 TRP 的总功率，在线性功率域除以
`tx_array.num_array_panels` 得到每物理面板功率；它不除以网络小区数、站点数、
TRP 数或 TXRU 数。旧 `trp.panel_power_mode` 仅作为被忽略的兼容输入保留。

`scheduler.max_mu_order: auto` 由 RF architecture 和调度域解析。修改阵列或 RF 逻辑时，
至少检查：

- `tests/test_array_config.py`
- `tests/test_scheduler_v210.py`
- `rf_architecture_summary.json`
- `array_config_summary.json`

## 7. Feedback 与调度

### 7.1 Feedback schemes

常用方案：

- `full_gamma`：上报完整/域内 Gamma，可预测共调度干扰；
- `baseline`：主要使用 SU 测量，不显式建模共调度干扰；
- `topk_conflict_id`：上报有限个强干扰 beam ID；
- `threshold_conflict_set`：按阈值构造冲突集合。

有限反馈方案使用 UE report 中可见的信息；调度器不能泄漏真实 post-scheduling
信道结果。

### 7.2 Evaluation matrix

`evaluation.matrix` 可以让一个 feedback scheme 与多个 algorithm 组合。每个组合有
独立 `case_id`，输出中的 `scheme` 通常表示 case ID，另有 `feedback_scheme` 和
`algorithm` 字段用于拆分。

### 7.3 调度域

归一化后的主要模式：

- `global`
- `per_site_joint`
- `per_sector_independent`

各调度域独立选择 UE/beam，但最终 schedule 会合并到同一 TTI；实际链路评估必须看到
其他 site/sector 已调度 beam 产生的真实干扰。

### 7.4 算法

支持：

- `greedy`
- `exhaustive`
- `hard_conflict_greedy`
- `adaptive_lambda_greedy`

optimized greedy 与 legacy/reference 路径需要保持回归一致。穷举路径包含候选排序、
零上界剪枝和 branch-and-bound。
hard-conflict greedy 每轮先比较候选加入当前集合后的边际效用。边际效用相同时，
若候选 MCS 相同且 SNR 跨度不超过 `scheduler.hard_conflict_snr_close_db`
（默认 1 dB），优先选择对当前候选池冲突影响较小者；SNR 跨度更大时优先高 SNR；
剩余平局按 UE ID、beam index 确定性排序。

所有算法都必须使用当前 `scheduler.objective`。PF 模式下，普通 greedy、
full-Gamma greedy、exhaustive 和 hard-conflict greedy 均使用
`predicted_rate / Tbar` 参与实际候选选择；不能只在最终统计
`objective_value` 时补乘 PF 权重。hard-conflict 的候选池元数据
`node_weight` 在 PF 模式下应为 `pf_weighted_su_rate`。

### 7.5 目标函数与 PF

`sum_rate` 使用预测速率和。

`proportional_fair` 每个 UE 维护历史平均实际 goodput `Tbar`，权重为：

```text
w_i[t] = 1 / max(Tbar_i[t], epsilon)
metric_i[t] = predicted_rate_i[t] * w_i[t]
```

MU 集合目标是对被选 UE 的 metric 求和。每个 TTI 更新：

```text
alpha = 1 / scheduler.pf_averaging_window_tti
Tbar_i[t+1] = (1-alpha)*Tbar_i[t] + alpha*actual_goodput_i[t]
```

未调度 UE 的 `actual_goodput=0`，也必须更新；调度权值是从 `Tbar` 派生的，不需要
另存一个独立静态权值。

## 8. 链路评估、MCS 与 OLLA

调度器侧预测和仿真真值必须分开：

- `predicted_sinr_db` / `predicted_mcs`：调度器可见信息；
- `effective_sinr_db`：合并 schedule 后用真实 H 和共调度干扰计算；
- EESM 把频域 SINR grid 映射为 effective SINR；
- actual MCS、TBLER、ACK/NACK 和 goodput 在链路评估阶段得到。

OLLA 为每个 `(case_id, UE)` 维护 backoff `b`：

```text
mcs_selection_sinr_db = predicted_sinr_db - b

ACK:  b <- b - delta
NACK: b <- b + delta*(1-target_bler)/target_bler
```

`effective_sinr_db` 只用于给已选 MCS 查询 TBLER/ACK，不参与 MCS 选择。这种不对称步长使期望更新量在目标 BLER 处为零。OLLA 在同一 drop 的 TTI 间保持，
新 drop 重置。warmup TTI 更新 OLLA，但不进入正式吞吐/BLER/CDF。

actual MCS 复用初始化阶段的 `SchedulerLinkLookup`：
`predicted_sinr_db - OLLA backoff -> actual MCS`，运行期不再逐 UE 调用 ILLA。
TBLER 与 MCS 选择是两条独立路径：

- 默认初始化 `PHYTblerLookup`，用正式 Sionna `PHYAbstraction` 批量生成
  `TBLER(SINR, MCS)` 网格，TTI 循环用 NumPy 线性插值；
- `link_abstraction.tbler_lookup.enabled=false` 时不构表，但同一 TTI 的全部已调度 UE
  仍合并为一次批量 `PHYAbstraction` 调用。

`link_abstraction_status.json` 记录 scheduler lookup 和 PHY TBLER lookup 的范围、
表尺寸及构建耗时；`runtime_phases.csv` 记录两者的初始化阶段。

## 9. 共同随机数与对比

- `system.random_seed` 与 drop 索引通过 `SeedSequence` 派生彼此独立的 drop RNG；
  每个 drop 依次用于 UE 撒点和 NumPy 信道随机量。Sionna 后端再从该 drop RNG
  派生 `sionna.phy.config.seed`，因此整次仿真可复现而不同 drop 不会复用同一随机流。
- evaluation case 尽量从相同 ACK 随机流开始，降低方案对比方差。
- baseline no-interference upper bound 复用 baseline schedule，并强制忽略波束间干扰。
- 上界参考的 ACK 随机数应与原 schedule 对齐。
- 比较不同方案时，先检查 schedule 是否相同，再解释 goodput 差异。

## 10. 主要输出及解读

输出目录通常为 `runs/<name>/`。

### 10.1 状态 JSON

- `rf_architecture_summary.json`：解析后的 RF 架构、TX unit 和并行波束能力；
- `array_config_summary.json`：阵列和码本解析结果；
- `link_abstraction_status.json`：实际使用的链路抽象后端及 fallback 状态；
- `sionna_import_probe.json`：Sionna 可用性诊断。

### 10.2 核心 CSV

| 文件 | 粒度与用途 |
|---|---|
| `metrics/link_tti.csv` | 每条已调度链路、每个正式 TTI；看调度预测 SNR/MCS、真实 SINR/MCS、TBLER、ACK、goodput、OLLA |
| `metrics/schedules.csv` | 调度结果；连续模式含每 TTI schedule |
| `metrics/ue_goodput.csv` | 每 `(drop, UE)` 跨全部正式 TTI 的平均 goodput；未调度 UE 计零 |
| `metrics/system_tti_goodput.csv` | 每 `(scheme, drop, TTI)` 的系统总 goodput；显式包含零吞吐 TTI |
| `metrics/system_drop_avg_goodput.csv` | 每 `(scheme, drop)` 在完整正式统计窗口内的平均系统 goodput |
| `metrics/summary.csv/json` | 方案级系统吞吐、UE 边缘吞吐、BLER、增益摘要 |
| `metrics/cell0_local_nack_rate.csv` | 小区 0、每 drop、每方案、每 UE 的局部 NACK rate；每个点严格包含 500 个该 UE 被调度的正式 TTI |
| `metrics/cell0_local_nack_rate_status.csv` | 小区 0 每 UE 的已调度样本数、完整 500-TTI 窗口数和未纳入曲线的尾段数 |
| `metrics/scheduler_stats.csv` | 调度器候选数、剪枝、耗时和最终调度规模 |
| `metrics/scheduler_iterations.csv` | greedy 逐轮选择信息 |
| `metrics/runtime_phases.csv` | feedback、scheduler、link evaluation 等阶段耗时 |
| `metrics/drops.csv` | 每 drop 网络规模、后端、TTI 模式和信道诊断 |
| `metrics/reports.csv` | UE feedback report JSON |
| `metrics/topk_interference_details.csv` | 每个 Top-K `(drop, UE, service beam, interferer beam)` 的服务 SNR/MCS、pair SINR/MCS、SINR/MCS 降幅和 outage；每 drop 测量一次 |
| `metrics/measurement_domains.csv` | 每 `(drop, UE)` 的服务 cell、测量域 cells、服务/测量/总上报波束数量 |
| `metrics/su_snr_samples.csv` | 所有上报候选的 SU-SNR |
| `metrics/su_snr_max_per_ue.csv` | 每 UE 最大上报 SU-SNR |
| `metrics/schedule_similarity*.csv` | 不同 case 的 schedule 集合相似度 |
| `metrics/scheduled_ue_su_throughput.csv` | 被调度 UE 在所选 beam 上的 standalone SU throughput |
| `metrics/gamma_measurement_backend.csv` | Gamma CPU/GPU 后端与耗时 |
| `metrics/channel_backend.csv` | 每 drop 实际信道后端和 fallback 信息 |

解读原则：

- 系统吞吐看 `summary`；
- 瞬时系统吞吐分布看 `system_tti_goodput_cdf.png`，drop 间短时平均分布看
  `system_drop_avg_goodput_cdf.png`；
- 5% UE 边缘吞吐和公平性看 `ue_goodput.csv`，不要只看已调度 UE；
- MCS/OLLA 问题看 `link_tti.csv` 中
  `predicted_sinr_db`、`predicted_mcs`、`actual_mcs`、`effective_sinr_db`、
  `mcs_selection_sinr_db`、`tbler`、`ack`、`olla_offset_db`；
- 调度复杂度看 `scheduler_stats.csv` 和 `runtime_phases.csv`；
- 方案增益异常时先看 `schedule_similarity`，排除“相同 schedule、仅 ACK 随机噪声”。

## 11. 测试地图

| 测试 | 主要覆盖 |
|---|---|
| `tests/test_smoke.py` | 主流程、链路、OLLA warmup、输出与基础工具 |
| `tests/test_array_config.py` | 阵列、panel、极化、码本和 Sionna 天线映射 |
| `tests/test_scheduler_v210.py` | optimized/legacy 调度一致性和资源约束 |
| `tests/test_sionna_link_adapter.py` | Sionna SYS 链路适配与标准 MCS/TBS |
| `tests/test_evaluation.py` | evaluation matrix 和 case |
| `tests/test_plot_cdf.py` | CDF 数据处理与绘图输入 |
| `tests/test_pf_continuous_tti.py` | PF EWMA 和连续 TTI 多普勒状态 |

推荐验证顺序：

```bash
python -m pytest tests/test_target.py -q --tb=short -x
python -m pytest tests/test_smoke.py -q --tb=short -x
python -m pytest -q --tb=short
```

仿真流程变化还应运行小规模 smoke configuration，检查输出字段和 TTI 数。

## 12. 按任务快速定位

| 任务 | 首先读取 | 常见测试/输出 |
|---|---|---|
| 修改 drop/TTI 循环 | `sim.py`, `channel.py`, `link.py` | `test_smoke.py`, `test_pf_continuous_tti.py`, `link_tti.csv` |
| 修改 PF/调度目标 | `scheduler.py`, `sim.py` | `test_scheduler_v210.py`, `schedules.csv`, `ue_goodput.csv` |
| 修改 OLLA/MCS | `link.py`, `link_adaptation.py`, `mcs.py` | `test_smoke.py`, `link_tti.csv` |
| 修改 Gamma/反馈 | `measurement.py`, `feedback.py`, `scheduler.py` | `su_snr_samples.csv`, `reports.csv`, `gamma_measurement_backend.csv` |
| 修改阵列/RF/TXRU | `codebook.py`, `rf.py`, `channel.py` | `test_array_config.py`, `rf_architecture_summary.json` |
| 修改 Sionna 信道 | `channel.py` | `test_array_config.py`, `channel_backend.csv`, import probe |
| 修改实验矩阵 | `evaluation.py`, `sim.py` | `test_evaluation.py`, `schedules.csv` |
| 修改输出统计 | `sim.py`, `plotting.py` | 目标 CSV、`summary.csv/json` |
| 修改 coverage | `coverage.py`, `plotting.py` | coverage CSV、heatmap figures |

## 13. 文档关系

- `README.md`：面向用户的主要运行和功能说明；
- `docs/yaml_parameter_reference.md`：完整 YAML 参数参考；
- `docs/implementation_notes.md`：实现约定和历史设计说明；
- `docs/mcs_tti_diagnostic.md`：MCS、TTI、OLLA 问题诊断；
- `更新说明v*.md` / `修改说明v*.md`：按版本记录目的、修改、用法和输出；
- 本文件：供后续代理快速建立代码上下文，不替代上述用户文档。
## v2.12 现行静态簇约定

此前“`measurement.domain_mode` 独立于 `scheduler.domain_mode`”的设计已被
替代。现行流程在信道生成后调用
`measurement.associate_ues_by_average_rsrp`，以最佳 TX/RX 波束对的频点平均
接收功率最大值关联 cell；随后
`topology.resolve_static_scheduling_clusters` 校验静态 cell 分区完整且不交叠，
`assign_ues_to_scheduling_clusters` 令 UE 唯一继承 serving cell 的簇。
`cluster_cell_ids_by_ue` 同时给出 Gamma 测量范围；`UEReport.scheduling_cluster`
是调度分组键。簇外波束不进入调度预测，合并 schedule 后的链路评估仍按真实
全网发射计算干扰。主要配置为 `scheduler.cluster_mode` 和
`scheduler.static_clusters`；旧 domain key 只用于兼容。

`metrics/runtime_phases.csv` 记录 drop 级
`topology_generation/channel_generation/average_rsrp_association/`
`static_cluster_preparation/gamma_measurement/feedback_generation`，以及 case 级
`scheduler/link_evaluation` 的 `elapsed_s`。这些阶段完成后也通过 progress
立即打印耗时。
