# Sionna SLS Beam Management Platform v2.4

本项目是一个面向“服务波束 + 干扰波束上报”的系统级波束管理仿真原型。默认场景为 **1-site 3-sector**，默认 TRP 天线为：

```text
4 TXRUs, 1024 AEs
(M, N, P, Mg, Ng; Mp, Np) = (16, 16, 2, 2, 1; 1, 1)
(dH, dV) = (0.5, 0.5)
```

v2.4 的核心变化是新增了 **RF architecture** 配置层，代码会自动把射频架构、波束发射方式和 MU order 关联起来：

- 情况 1：`panel_polarization_subarray`，即 sub-connected / panel-polarization connected；
- 情况 2：`fully_connected`，即 fully-connected hybrid beamforming；
- 默认参数为情况 1，允许不同极化采用不同波束，每个物理面板/极化子阵列独立发射 DFT 波束；
- 默认 `scheduler.max_mu_order: auto`，会根据 RF architecture 自动解析；
- 默认 4 TXRUs，因此默认最大同时发射模拟波束数 = 4，默认最大 MU order = 4；
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

---

## 2. 默认 topology

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

当前版本默认仍是 1-site 3-sector，没有实现 7-site / 21-sector wrap-around 动态邻区干扰。

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

```text
num_ae = M * N * P * Mg * Ng * Mp * Np
       = 16 * 16 * 2 * 2 * 1 * 1 * 1
       = 1024
```

DFT 空间码本不乘极化数 `P`：

```text
H = N * Ng * Np = 16
V = M * Mg * Mp = 32
full spatial codebook size = H * V = 512
beam vector length = H * V * P = 1024 AEs
```

默认 SLS 扫描不是完整 512 个方向，而是均匀采样：

```yaml
tx_array:
  num_beams_h: 4
  num_beams_v: 4
  max_beams: 16
```

即每个活动码本扫描：

```text
num_beams_h * num_beams_v = 16 beams
```

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

```text
max_parallel_beams_per_trp = num_txru = 4
scheduler.max_mu_order(auto) = 4
```

在默认 1-site 3-sector、每 sector 1 个 TRP 的情况下：

```text
每 sector: 4 TX units * 16 beams = 64 TX beam IDs
全网: 3 sectors * 64 = 192 TX beam IDs
```

### 4.2 情况 1 但同一面板两个极化共享 beam

配置：

```yaml
rf_architecture:
  txru_connectivity: panel_polarization_subarray
  allow_independent_polarization_beams: false
```

含义：同一 panel 的两个极化共享一个空间 beam，不允许两个极化独立扫不同方向。默认 TRP 有 2 个物理面板，因此：

```text
max_parallel_beams_per_trp = number_of_physical_panels = 2
scheduler.max_mu_order(auto) = 2
```

### 4.3 情况 2：fully-connected hybrid beamforming

配置：

```yaml
rf_architecture:
  txru_connectivity: fully_connected
  num_txru: 4
```

含义：每个 TXRU 都连接到整个 TRP 的 1024 AEs，每个 TXRU 可形成一个 full-array DFT beam。因此：

```text
max_parallel_beams_per_trp = num_txru = 4
scheduler.max_mu_order(auto) = 4
```

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
  algorithm: greedy
  objective: sum_rate
  max_mu_order: auto
```

`greedy` 推荐用于常规仿真；`exhaustive` 只建议用于很小的 debug 配置，因为复杂度会随 UE 数、候选 beam 数和 MU order 组合爆炸。

`max_mu_order: auto` 时，程序根据 RF architecture 自动设置最大同时调度 UE 数。若手动写整数，例如：

```yaml
scheduler:
  max_mu_order: 3
  cap_mu_order_by_rf: true
```

实际 MU order 会被 RF 物理并发 beam 数截断：

```text
effective_max_mu_order = min(3, max_parallel_beams_per_trp)
```

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
