# YAML 参数说明文档（v2.4，自包含）

本文档说明 `configs/v2_one_site_three_sector.yaml` 中主要可配置参数的含义、取值范围和注意事项。

---

## 1. `scenario`

| 参数 | 含义 | 典型取值 / 范围 | 说明 |
|---|---|---|---|
| `name` | 场景名称 | 字符串 | 仅用于输出标识。 |
| `carrier_frequency_ghz` | 载波频率，单位 GHz | 正数，例如 `3.5`, `7`, `30` | 影响 pathloss、Sionna TR 38.901 channel 和阵列物理尺度解释。 |
| `channel_model` | 信道 backend | `sionna_tr38901_uma`, `sionna_tr38901_umi`, `sionna_tr38901_rma`, `numpy_geometric_uma` | 优先使用 Sionna TR 38.901；若不可用且 `sionna.fallback_to_numpy_if_unavailable=true`，会 fallback 到 numpy 几何信道。 |
| `min_ue_distance_m` | UE 与站点最小距离 | `>0` | UE drop 和覆盖图会避开过近点。 |
| `max_ue_distance_m` | UE 与站点最大距离 | `> min_ue_distance_m` | UE drop 半径上限。 |
| `enable_pathloss` | 是否启用 pathloss | `true`/`false` | 对 Sionna backend 和 fallback backend 都有意义。 |
| `enable_shadow_fading` | 是否启用阴影衰落 | `true`/`false` | 对 Sionna backend 和 fallback backend 都有意义。 |
| `o2i_model` | UMa/UMi O2I 模型 | `low`, `high` | 传给 Sionna UMa/UMi。 |
| `num_clusters` | fallback 多径簇数 | 正整数 | 仅 `numpy_geometric_uma` 或 fallback 使用。 |
| `delay_spread_ns` | fallback RMS delay spread 近似 | 正数，单位 ns | 仅 fallback 使用。 |
| `shadow_fading_std_db` | fallback 阴影衰落标准差 | 非负数，单位 dB | 仅 fallback 使用。 |
| `pathloss_exponent` | fallback pathloss 指数 | 通常 `2~4` | 仅 fallback 使用。 |

---

## 2. `topology`

| 参数 | 含义 | 典型取值 / 范围 | 说明 |
|---|---|---|---|
| `layout` | 拓扑布局 | `one_site_three_sector`, `three_site_triangle`, `seven_site_hex` | 也支持别名 `multi_site` + `num_sites` in `{1,3,7}`。 |
| `num_sites` | 站点数 | `1`, `3`, `7` | 多站点布局会按该字段校验站点数。 |
| `sectors_per_site` | 每站 sector 数 | 默认 `3` | 默认 3 sector/cell。 |
| `sector_azimuths_deg` | 每个 sector 的方位角 | 长度为 `sectors_per_site` 的数组 | 默认 `[30,150,270]`。 |
| `sector_width_deg` | sector 宽度 | 通常 `120` | 用于 UE drop 和 topology 图。 |
| `isd_m` | 站间距 | 正数，单位 m | 三站点中任意两站点间距为 `isd_m`；七站点中中心站到第一圈邻站为 `isd_m`。 |
| `bs_height_m` | 基站高度 | 正数，单位 m | 传给 topology/channel。 |

输出 `figures/topology.png` 会画出站点、sector、UE 和 ISD 标尺。

---

## 3. `system`

| 参数 | 含义 | 典型取值 / 范围 | 说明 |
|---|---|---|---|
| `subcarrier_spacing_khz` | SCS | `15/30/60/120` 等 | 与 `pdsch.num_prbs` 一起唯一确定有效占用带宽。 |
| `tx_power_dbm` | 发射功率 | dBm | 默认按 TX unit 平均分配。 |
| `num_drops` | 随机 drop 数 | 正整数 | 越大统计越稳定，仿真越慢。 |
| `num_tti_per_drop` | 每个 drop 的 TTI 数 | 正整数 | OLLA/HARQ 风格随机 ACK 统计使用。 |
| `random_seed` | 随机种子 | 整数 | 用于复现实验。 |
| `target_bler` | 目标 BLER | `(0,1)`，默认 `0.1` | ILLA/MCS 选择和 OLLA 使用。 |

---

## 4. `pdsch`

| 参数 | 含义 | 典型取值 / 范围 | 说明 |
|---|---|---|---|
| `num_prbs` | PRB 数 | 正整数 | 用于 TBS/goodput，并与 SCS 一起确定信道频率跨度和噪声带宽。 |
| `num_symbols` | PDSCH OFDM symbols | `1~14` | 默认 `12`。 |
| `dmrs_overhead_re_per_prb` | 每 PRB DMRS 开销 RE | 非负整数 | 用于 TBS 近似。 |
| `num_layers_per_ue` | 每 UE 层数 | 正整数 | 当前主流程默认 single-layer。 |
| `slot_duration_ms` | slot 时长 | 正数，单位 ms | 用于 Mbps 换算。 |

有效占用带宽不再单独配置，而是统一计算为：

\[
B_{\mathrm{occupied}} =
\mathrm{num\_prbs}\times 12\times
\mathrm{subcarrier\_spacing\_khz}\times 10^3.
\]

例如 `132 PRB × 12 × 120 kHz = 190.08 MHz`。该值同时用于信道频率采样跨度和
热噪声积分带宽，并写入 `resolved_config.yaml` 的
`_resolved.occupied_bandwidth_mhz` 以及 `metrics/drops.csv`。

---

## 5. `noise`

| 参数 | 含义 | 典型取值 / 范围 | 说明 |
|---|---|---|---|
| `thermal_noise_density_dbm_per_hz` | 热噪声谱密度 | 通常 `-174` | dBm/Hz。 |
| `ue_noise_figure_db` | UE 噪声系数 | 通常 `5~9` dB | 默认 `7` dB。 |

---

## 6. `ue_drop`

| 参数 | 含义 | 典型取值 / 范围 | 说明 |
|---|---|---|---|
| `num_ut_per_sector` | 每 sector UE 数 | 正整数 | 默认 10，因此 1-site 3-sector 下总 UE 数 30。 |
| `distribution` | UE 分布 | 当前默认 `uniform_in_sector` | 在 sector 扇形区域内均匀 drop。 |
| `speed_kmh` | UE 速度 | 非负数 | 传给 Sionna topology；fallback 中影响较小。 |

---

## 7. `trp`

| 参数 | 含义 | 典型取值 / 范围 | 说明 |
|---|---|---|---|
| `num_trps_per_sector` | 每 sector TRP 数 | 正整数，默认 `1` | v2.4 推荐使用此字段。 |
| `num_panels_per_sector` | 兼容旧字段 | 正整数 | 建议与 `num_trps_per_sector` 保持一致；RF 架构才决定每 TRP 有多少 TX units。 |
| `panel_azimuth_offsets_deg` | 物理阵列面板相对 sector boresight 的方位偏置 | 数组，单位 deg | 对 sub-connected 架构，不同物理面板可给不同 offset。默认 `[0,0]`。 |
| `panel_power_mode` | 功率分配方式 | `per_tx_unit_equal`, `per_panel_equal`, `per_txru_equal`, `total_per_tx_unit` | 默认 `per_tx_unit_equal`，总功率除以全网 TX units。非 equal 模式会把 `tx_power_dbm` 作为每 TX unit 功率。 |

---

## 8. `rf_architecture`

v2.4 新增的关键配置块。

| 参数 | 含义 | 取值范围 | 说明 |
|---|---|---|---|
| `txru_connectivity` | TXRU 与 AEs 的连接结构 | `panel_polarization_subarray` 或 `fully_connected` | 情况 1 或情况 2。支持别名：`sub_connected`, `case1`, `fully_connected_hybrid`, `case2` 等。 |
| `allow_independent_polarization_beams` | 同一物理面板的两个极化是否可用不同 beam | `true`/`false` | 仅对 `panel_polarization_subarray` 关键。默认 `true`。 |
| `num_txru` | TRP TXRU 数 | 正整数 | 默认 4。通常应与 `tx_array.num_txru` 一致。 |
| `max_parallel_beams_per_trp` | 每 TRP 最大并发模拟 beam 数 | 当前建议 `auto` | 解析结果写入 `rf_architecture_summary.json`。 |

### 8.1 情况 1：`panel_polarization_subarray`

默认：

```yaml
rf_architecture:
  txru_connectivity: panel_polarization_subarray
  allow_independent_polarization_beams: true
  num_txru: 4
```

默认 TRP 有：

$$
\mathrm{physical\ panels} = M_g \times N_g \times M_p \times N_p = 2
$$

$$
\mathrm{polarizations} = P = 2
$$

$$
\mathrm{panel\ polarization\ TX\ units} = 2 \times 2 = 4
$$

因此：

$$
\mathrm{max\_parallel\_beams\_per\_trp} = 4
$$

$$
\mathrm{scheduler.max\_mu\_order(auto)} = 4
$$

每个 TXRU 只使用对应 panel-polarization 子阵列的 DFT beam。

如果：

```yaml
allow_independent_polarization_beams: false
```

则同一面板两个极化共享一个空间 beam：

$$
\mathrm{max\_parallel\_beams\_per\_trp} = \mathrm{physical\ panels} = 2
$$

$$
\mathrm{scheduler.max\_mu\_order(auto)} = 2
$$

### 8.2 情况 2：`fully_connected`

配置：

```yaml
rf_architecture:
  txru_connectivity: fully_connected
  num_txru: 4
```

含义：每个 TXRU 连接到全 TRP 阵列，可独立形成一个 full-array DFT beam：

$$
\mathrm{max\_parallel\_beams\_per\_trp} = \mathrm{num\_txru} = 4
$$

$$
\mathrm{scheduler.max\_mu\_order(auto)} = 4
$$

注意：这个模式隐含 fully-connected 或足够灵活的 hybrid RF 网络；如果实际硬件是 sub-connected，则 full-array 多 beam 不成立。

---

## 9. `tx_array`

| 参数 | 含义 | 取值范围 | 说明 |
|---|---|---|---|
| `model` | 阵列参数模式 | `tr38901_panel` | 使用 3GPP-style 参数。 |
| `num_txru` | TXRU 数 | 正整数 | 与 `rf_architecture.num_txru` 一致。 |
| `num_ae` | AE 数 | 正整数 | 程序校验 `M*N*P*Mg*Ng*Mp*Np`。 |
| `M` | 每 panel 垂直单元数 | 正整数 | 默认 16。 |
| `N` | 每 panel 水平单元数 | 正整数 | 默认 16。 |
| `P` | 极化数 | `1` 或 `2` 常用 | 默认 2。 |
| `Mg` | 垂直方向 panel 数 | 正整数 | 默认 2。 |
| `Ng` | 水平方向 panel 数 | 正整数 | 默认 1。 |
| `Mp` | 每 panel group 垂直 repetition | 正整数 | 默认 1。 |
| `Np` | 每 panel group 水平 repetition | 正整数 | 默认 1。 |
| `dH` | 水平阵元间距 | 正数，单位 wavelength | 默认 0.5。 |
| `dV` | 垂直阵元间距 | 正数，单位 wavelength | 默认 0.5。 |
| `beam_scope` | legacy/manual beam scope | `joint`, `per_panel` | v2.4 主流程会由 RF architecture 解析有效 scope；该字段主要用于兼容和 UE/codebook fallback。 |
| `sampling_mode` | DFT beam 采样方式 | `uniform`, `centered` | 默认均匀采样。 |
| `num_beams_h` | 水平扫描 beam 数 | 正整数，不超过活动码本水平大小 | 默认 4。 |
| `num_beams_v` | 垂直扫描 beam 数 | 正整数，不超过活动码本垂直大小 | 默认 4。 |
| `max_beams` | 最大扫描 beam 数上限 | `null` 或正整数 | 通常设为 `num_beams_h*num_beams_v`。 |
| `vertical_beam_mode` | 垂直 beam 模式 | `scan` 或 `fixed` | `fixed` 时只用 `fixed_v_index`。 |
| `fixed_v_index` | 固定垂直 DFT index | `null` 或整数 | 用作电下倾角候选。 |

码本大小：

$$
\mathrm{full\ array\ DFT\ spatial\ codebook}
= N \times N_g \times N_p \times M \times M_g \times M_p
$$

$$
\mathrm{per\ panel\ DFT\ spatial\ codebook} = N \times M
$$

默认 sub-connected/panel-polarization 模式下，每个 TXRU 使用对应 physical panel 的 local DFT codebook，即 `N*M = 256` 个完整候选方向；默认 SLS 只采样 `4*4=16` 个。

---

## 10. `ue_array`

UE 阵列字段与 `tx_array` 基本一致，只是 `num_rxru` 替代 `num_txru`。默认：

```yaml
ue_array:
  num_rxru: 4
  num_ae: 16
  M: 4
  N: 4
  P: 1
```

即 4×4 单极化 UE 阵列，16 AEs，默认扫描 4×4 = 16 个 RX DFT beams。

---

## 11. `beam`

| 参数 | 含义 | 取值范围 | 说明 |
|---|---|---|---|
| `tx_codebook` | TX 码本类型 | 当前 `dft_2d` | 所有 TX beams 使用 DFT beam。 |
| `rx_codebook` | RX 码本类型 | 当前 `dft_2d` | 所有 RX beams 使用 DFT beam。 |
| `one_beam_per_panel` | 每 TX unit 同时至多一个 beam | `true`/`false` | 默认 `true`。调度约束实际通过 `scheduler.use_panel_constraint` 生效。 |

---

## 12. `measurement`

| 参数 | 含义 | 取值范围 | 说明 |
|---|---|---|---|
| `num_freq_points` | 频域采样点数 | 正整数 | Gamma 矩阵和 EESM 使用。越大越慢。 |
| `compute_full_gamma` | 是否计算完整服务/干扰 beam Gamma | `true`/`false` | 当前主流程需要 Gamma 用于 full_gamma/oracle；实际计算会按 `scheduler.domain_mode` 裁剪到 UE 的调度域。 |
| `frequency_average` | 频域平均方式 | 当前 `linear_power` | 预留字段。 |

复杂度大致随：

$$
N_{\mathrm{UE}} \times N_{\mathrm{TX\ beam}}^2
\times N_{\mathrm{RX\ beam}} \times N_{\mathrm{freq}}
$$

增长。

---

## 13. `feedback`

| 参数 | 含义 | 取值范围 | 说明 |
|---|---|---|---|
| `schemes` | 反馈方案列表 | `full_gamma`, `baseline`, `topk_conflict_id`, `threshold_conflict_set` | 可选择一个或多个。 |
| `cqi_mode` | CQI/MCS 选择方式 | `illa_target_bler` | 通过 link adapter 选择 MCS。 |
| `service_beam_top_k1` | UE 上报服务 beam 数 | 正整数 | baseline/proposed 使用。 |
| `oracle_service_beam_top_k` | full_gamma/oracle 候选服务 beam 数 | 正整数 | 越大越慢。 |
| `conflict_top_k2` | 每个服务 beam 上报强干扰 beam 数 | 非负整数 | `topk_conflict_id` 使用。 |
| `conflict_sinr_threshold_db` | 干扰冲突阈值 | dB | `threshold_conflict_set` 使用。 |

---

## 14. `scheduler`

| 参数 | 含义 | 取值范围 | 说明 |
|---|---|---|---|
| `domain_mode` | 调度域模式 | `single_site_three_sector_independent`, `per_site_joint`, `global` | 默认 `per_site_joint`。`single_site_three_sector_independent` 表示每个 sector 独立测量和调度；`per_site_joint` 表示同站点 3 个 sector 联合测量和调度。 |
| `objective` | 优化目标 | `sum_rate`, `proportional_fair` | 总吞吐最大或比例公平。 |
| `max_mu_order` | 最大 MU order | `auto` 或正整数 | `auto` 时由 RF architecture 自动解析。 |
| `cap_mu_order_by_rf` | 是否用 RF 物理并发 beam 数截断手动 MU order | `true`/`false` | 默认 `true`。 |
| `algorithm` | 调度算法 | `greedy`, `exhaustive`, `hard_conflict_greedy`, `adaptive_lambda_greedy` | `hard_conflict_greedy` 在候选二元组级实施硬冲突删除；`adaptive_lambda_greedy` 使用场景自适应冲突惩罚。 |
| `use_panel_constraint` | 是否约束同一 TX unit 同时只选一个 beam | `true`/`false` | 默认 `true`。 |
| `exhaustive_pruning.enabled` | 是否启用穷举剪枝配置 | `true`/`false` | 默认 `true`。 |
| `exhaustive_pruning.sort_by_upper_bound` | 是否按单用户上界排序 | `true`/`false` | 先找到较好当前最优值，帮助上界剪枝。 |
| `exhaustive_pruning.zero_upper_bound` | 是否移除零上界 UE report | `true`/`false` | 不改变最优性。 |
| `exhaustive_pruning.branch_and_bound` | 是否启用 branch-and-bound 上界剪枝 | `true`/`false` | 不改变当前候选集合内的穷举最优性。 |
| `conflict_penalty_lambda` | proposed feedback 冲突惩罚权重 | 非负数 | 用于 ID-only conflict penalty。 |
| `conflict_penalty_mode` | lambda 模式 | `fixed`, `adaptive` | `adaptive` 时忽略固定值，按候选 SU rate 中位数计算。 |
| `adaptive_lambda_alpha` | 自适应 lambda 比例 | 非负数，常用 `0.1`, `0.2`, `0.5` | `lambda = alpha * median(candidate SU rate [Mbps])`，逐调度域计算。 |
| `unknown_interference_policy` | 未知干扰处理 | 当前 `zero` | 预留字段。 |
| `pf_tbar_init_mbps` | PF 初始平均吞吐 | 正数 | `objective=proportional_fair` 时使用。 |

`max_mu_order:auto` 的解析规则：

$$
\mathrm{panel\_polarization\_subarray + independent\ polarization\ beams}:
\quad \mathrm{max\_mu\_order} = \mathrm{num\_txru}
$$

$$
\mathrm{panel\_polarization\_subarray + shared\ polarization\ beam}:
\quad \mathrm{max\_mu\_order} = \mathrm{physical\ panel\ count}
$$

$$
\mathrm{fully\_connected}:
\quad \mathrm{max\_mu\_order} = \mathrm{num\_txru}
$$

如果 `num_trps_per_sector > 1`，则每 sector 的 RF 并发上限会乘以 TRP 数。

---

## 15. `link_abstraction`

| 参数 | 含义 | 取值范围 | 说明 |
|---|---|---|---|
| `mode` | 链路抽象模式 | `sionna_sys_precomputed_bler` | 优先 Sionna SYS；不可用则 fallback。 |
| `mcs_table_index` | MCS table index | 整数 | 传给 Sionna SYS 或 fallback 记录；下行 NR PDSCH table 1 用 `1`。 |
| `mcs_category` | MCS category | 整数 | Sionna SYS 中 `1=PDSCH`、`0=PUSCH`；本项目下行 PDSCH 默认用 `1`。 |
| `sinr_mapping` | SINR mapping | 当前 `eesm` | 频选 SINR 到有效 SINR。 |
| `eesm_beta_db` | EESM beta | 正数，dB | 默认 5 dB。 |
| `olla_enabled` | 是否启用 OLLA | `true`/`false` | 默认 `true`。 |
| `olla_step_db` | OLLA 步长 | 正数，dB | 默认 0.1 dB。 |
| `olla_warmup_tti` | 每个 drop 的 OLLA 预热 TTI 数 | 非负整数 | 默认 `0`。预热期间正常抽样 ACK、更新 MCS/OLLA，但不写入 `link_tti.csv`，也不进入吞吐、BLER、CDF 等统计；`system.num_tti_per_drop` 始终只表示正式统计 TTI 数。 |
| `harq_enabled` | HARQ 预留开关 | `true`/`false` | 当前为预留/记录字段。 |
| `bler_curve_slope` | fallback logistic BLER 斜率 | 正数 | 仅 fallback 使用。 |
| `fallback_snr_min_db` | fallback table 最小 SNR | dB | 仅 fallback 使用。 |
| `fallback_snr_max_db` | fallback table 最大 SNR | dB | 仅 fallback 使用。 |
| `fallback_snr_step_db` | fallback table SNR 步长 | 正数 dB | 仅 fallback 使用。 |

下行 PDSCH、目标 BLER 10% 的推荐配置：

```yaml
system:
  target_bler: 0.1

link_abstraction:
  mode: sionna_sys_precomputed_bler
  mcs_table_index: 1
  mcs_category: 1
  olla_warmup_tti: 100
```

---

## 16. `coverage_heatmap`

| 参数 | 含义 | 取值范围 | 说明 |
|---|---|---|---|
| `enabled` | 是否生成覆盖图 | `true`/`false` | 大规模时可关闭加速。 |
| `backend` | 覆盖图 backend | `configured_channel_sampling` | 使用当前配置的 channel backend 对栅格采样。 |
| `grid_size` | 栅格边长点数 | 正整数 | 总点数约 `grid_size^2`，越大越慢。 |
| `max_distance_m` | 覆盖图最大半径 | 正数 m | 超出半径点忽略。 |
| `chunk_size` | 栅格分块大小 | 正整数 | 降低内存峰值。 |

### `fixed_vertical_beam_cdf`

| 参数 | 含义 | 取值范围 | 说明 |
|---|---|---|---|
| `enabled` | 是否生成固定垂直 beam CDF | `true`/`false` | 用于选电下倾角。 |
| `candidate_v_indices` | 候选垂直 DFT indices | `all` 或整数数组 | `all` 会遍历活动码本全部垂直 index。 |
| `horizontal_num_beams` | 每个垂直 beam 下扫描的水平 beam 数 | 正整数 | 对水平 beam 的 RSRP 做平均。 |
| `selection_metric` | 选择电下倾角的指标 | `mean_dbm`, `p05_dbm`, `p50_dbm`, `p95_dbm` | 默认选择平均覆盖 RSRP 最大的垂直 beam。 |

输出：

```text
figures/fixed_vertical_beam_cdf.png
metrics/fixed_vertical_beam_summary.csv
metrics/fixed_vertical_beam_samples.csv
fixed_vertical_beam_selection.json
```

---

## 17. `progress`

| 参数 | 含义 | 取值范围 | 说明 |
|---|---|---|---|
| `enabled` | 是否打印仿真进度 | `true`/`false` | 默认开启。也可用命令行 `--quiet` 关闭。 |

---

## 18. `sionna`

| 参数 | 含义 | 取值范围 | 说明 |
|---|---|---|---|
| `enable_import_probe` | 是否记录 Sionna 模块导入状态 | `true`/`false` | 输出 `sionna_import_probe.json`。 |
| `prefer_sionna_sys_phy_abstraction` | 是否优先 Sionna SYS | `true`/`false` | 不可用则 fallback。 |
| `fallback_to_numpy_if_unavailable` | Sionna channel 不可用时是否 fallback | `true`/`false` | 默认 `true`，保证代码可跑。 |
| `device` | Sionna/Torch device | `null` 或 device 字符串 | 默认 `null`。 |
| `precision` | Sionna 精度 | `null`, `single`, `double` 等 | 取决于本地 Sionna API。 |
| `bs_polarization` | BS 极化 | `single`, `dual` | 默认 `dual`。 |
| `bs_polarization_type` | BS 极化类型 | `V`, `H`, `VH`, `cross` 等 | 取决于 Sionna PanelArray。 |
| `bs_antenna_pattern` | BS 天线方向图 | `38.901`, `omni` 等 | 默认 `38.901`。 |
| `ut_polarization` | UE 极化 | `single`, `dual` | 默认 `single`。 |
| `ut_polarization_type` | UE 极化类型 | `V` 等 | 默认 `V`。 |
| `ut_antenna_pattern` | UE 天线方向图 | `omni`, `38.901` 等 | 默认 `omni`。 |

---

## 19. 推荐调试配置

为了快速确认程序逻辑，建议：

```bash
/home/zhangwei/anaconda3/envs/tf_sionna_rt/bin/python -m beam_sls.run \
  --config configs/v2_one_site_three_sector.yaml \
  --out runs/debug \
  --num-drops 1 \
  --num-tti 1 \
  --algorithm greedy \
  --skip-heatmap
```

如果需要看固定垂直 beam CDF，再打开覆盖图，但建议先降低：

```yaml
coverage_heatmap:
  grid_size: 20
  fixed_vertical_beam_cdf:
    candidate_v_indices: [0, 4, 8, 12]
```

---

## 20. 输出诊断文件

运行后重点检查：

```text
rf_architecture_summary.json
array_config_summary.json
link_abstraction_status.json
sionna_import_probe.json
metrics/beams.csv
metrics/summary.csv
metrics/ue_goodput.csv
metrics/schedule_similarity.csv
metrics/schedule_similarity_by_drop.csv
metrics/su_snr_samples.csv
metrics/su_snr_max_per_ue.csv
metrics/su_snr_summary.csv
metrics/scheduled_ue_su_throughput.csv
metrics/scheduled_ue_su_throughput_summary.csv
figures/scheduled_ue_su_throughput_cdf.png
metrics/paired_case_debug.txt
```

`rf_architecture_summary.json` 是确认 MU order 与射频架构是否按预期解析的核心文件。

`scheduled_ue_su_throughput.csv` 只包含最终被调度的 UE。每个样本使用所选 service
beam 的 standalone SNR/MCS 换算 SU 吞吐，SU outage 记为 0 Mbps；它不等同于
调度器预测的 MU rate，也不等同于 `link_tti.csv` 中含真实多用户干扰和 ACK/NACK
后的 goodput。

需要核对“相同 schedule 为什么得到不同 goodput”时，可配置：

```yaml
analysis:
  paired_case_debug:
    enabled: true
    pairs:
      - [baseline__greedy, threshold_conflict_set__greedy]
```

终端只需复制从 `[paired-debug] begin` 到 `[paired-debug] end` 的内容。
`link_tti.csv` 会同时记录 `ack_random_uniform` 和 `link_position`，用于判断共同随机数
是否因 link 遍历顺序不同而分配给了不同 UE。
