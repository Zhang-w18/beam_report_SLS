# MCS / TTI 问题诊断记录

日期：2026-06-13

## 1. 问题

从 `link_tti.csv` 里看到几个现象：

- 固定 `drop` 和 `ue_id` 后，`actual_mcs` 会随 `tti` 变化；
- `predicted_mcs` 经常大于 `actual_mcs`，例如 `27` 对 `20`，甚至出现很低的 `actual_mcs`；
- 有些结果里 `effective_sinr_db` 很小或非有限，但 `tbler=0` 且 `ack=1`；
- 需要确认：链路层到底应该用哪个 MCS 查 BLER。

## 2. 正确的链路逻辑

这里应该区分两类量：

```text
调度器侧可见量：预测 SINR / predicted_mcs / OLLA offset
仿真真值量：真实 post-scheduling effective_sinr_db
```

调度器不知道真实 `effective_sinr_db`，所以不能用真实等效 SINR 反过来选择传输 MCS。

更合理的流程是：

```text
1. 调度器根据有限上报信息得到 predicted_sinr_db 和 predicted_mcs；
2. OLLA 用 ACK/NACK 历史维护一个 backoff；
3. 本次传输 MCS = select_mcs(predicted_sinr_db - olla_offset_db)；
4. 仿真器用真实 effective_sinr_db 和本次传输 MCS 查询 TBLER；
5. 根据 TBLER 抽样 ACK/NACK；
6. ACK 让 backoff 下降，NACK 让 backoff 上升。
```

因此：

- `predicted_mcs` 是调度器输出的原始 MCS 估计；
- `actual_mcs` 是本次真正用于传输、查 TBLER、算 TBS/goodput 的 MCS；
- `effective_sinr_db` 是真实链路质量，只用于查 BLER/ACK，不用于选择 MCS；
- 新增的 `mcs_selection_sinr_db` 是 `predicted_sinr_db - olla_offset_db`，用于解释 `actual_mcs` 为什么变化。

`actual_mcs` 随 TTI 变化是正常的，因为 OLLA offset 会根据前面 TTI 的 ACK/NACK 更新。即使调度器选的波束和真实 `effective_sinr_db` 不变，MCS 也可能因为 OLLA 而变化。

## 3. 本次修复

### 3.1 修复 MCS 选择的因果性

旧逻辑中，`actual_mcs` 用真实 `effective_sinr_db + OLLA` 选择。这等于把调度器不知道的真实信道信息泄漏给 MCS 选择器。

已改为：

```text
actual_mcs = select_mcs(predicted_sinr_db - olla_offset_db)
tbler      = BLER_TABLE_LOOKUP(effective_sinr_db, actual_mcs)
```

修改文件：

```text
beam_sls/link.py
```

同时在 `link_tti.csv` 中新增：

```text
mcs_selection_sinr_db
```

这列用于记录实际选 MCS 时使用的调度器侧 SINR。

### 3.2 修复 Sionna SYS 不可用 BLER 表项被夹成 0 的问题

这里的“不可用 BLER 表项”指的是：Sionna SYS 的 BLER 表中没有该组合对应的有效 BLER 数值。

组合大致包括：

```text
mcs_category
mcs_table_index
mcs_index
code block size
SNR/SINR 插值网格点
```

如果某个组合不在 Sionna 的表里，Sionna 内部插值表会用 `inf` 表示“这个 BLER 表项不可用”。在修复前项目误命中的 `PUSCH_table1.json` 里，某些很低的 MCS index 没有有效表项。例如诊断时见到 `mcs_index=2` 对应不可用项。

旧代码的问题是：

```python
np.clip(tbler, 0.0, 1.0)
```

当 Sionna 返回原始 `tbler=-inf` 时，`np.clip(-inf, 0, 1)` 会变成 `0`。于是 CSV 里会出现不合理结果：

```text
effective_sinr_db 很低
actual_mcs 很低
Sionna 原始 tbler = -inf
项目写出的 tbler = 0
ack = 1
```

这不是物理上“低 SINR 也能可靠接收”，而是 `-inf` 被错误夹成了 `0`。

已修复为：

```text
如果 Sionna TBLER 是非有限值、负值，统一按 1.0 处理
```

也就是不可用表项按“本次传输必失败/不可可靠解码”处理，而不是按 `0` 错误通过。

修改文件：

```text
beam_sls/link_adaptation.py
```

### 3.2.1 Sionna BLER 表长什么样

本机 Sionna 默认 BLER 表目录：

```text
/Users/zhangwei/Downloads/lls_platform_sc_mimo/.venv-sionna1/lib/python3.10/site-packages/sionna/sys/bler_tables
```

目录下有 6 个 JSON 文件：

```text
PUSCH_table1.json
PUSCH_table2.json
PDSCH_table1.json
PDSCH_table2.json
PDSCH_table3.json
PDSCH_table4.json
```

Sionna `PHYAbstraction()` 默认会全部加载。文件和 category/index 的对应关系如下：

```text
文件                 category  index  MCS 范围
PUSCH_table1.json    0         1      3..27
PUSCH_table2.json    0         2      9..27
PDSCH_table1.json    1         1      3..28
PDSCH_table2.json    1         2      2..27
PDSCH_table3.json    1         3      9..28
PDSCH_table4.json    1         4      2..26
```

修复前项目配置是：

```yaml
link_abstraction:
  mcs_table_index: 1
  mcs_category: 0
```

因此修复前实际用的是：

```text
PUSCH_table1.json
category = 0
index = 1
```

本项目是下行 PDSCH，已改为：

```yaml
link_abstraction:
  mcs_table_index: 1
  mcs_category: 1
```

这对应 Sionna 的：

```text
PDSCH_table1.json
category = 1
index = 1
```

目标 BLER 10% 不对应另一张 JSON 表，而是 ILLA 的选择门限：

```yaml
system:
  target_bler: 0.1
```

ILLA 会在 `PDSCH_table1.json` 的 BLER 曲线上查找“在当前 SINR / code block size 下，TBLER 不超过 0.1 的最高 MCS”。

`PUSCH_table1.json` 的 JSON 层级大致如下：

```json
{
  "_comment": "BLER table for PUSCH channel, Table 1",
  "category": {
    "0": {
      "index": {
        "1": {
          "MCS": {
            "3": {
              "SNR_db": [-5.0, -3.2142857142857144, "...", 20.0],
              "EbN0_db": ["..."],
              "CBS": {
                "24": {
                  "BLER": [0.9283333420753479, 0.6100000143051147, "...", 0.0]
                },
                "100": {"BLER": ["..."]},
                "500": {"BLER": ["..."]},
                "1000": {"BLER": ["..."]},
                "2000": {"BLER": ["..."]}
              }
            },
            "4": {"...": "..."},
            "27": {"...": "..."}
          }
        }
      }
    }
  }
}
```

对修复前误用的 `PUSCH_table1.json`：

```text
可用 MCS: 3, 4, ..., 27
每个 MCS 的原始 SNR 点数: 15
原始 SNR 范围: -5.0 dB 到 20.0 dB
原始 CBS 点: 24, 100, 500, 1000, 2000
```

几个代表 MCS 的摘要：

```text
MCS  SNR范围(dB)  SNR点数  CBS点数  CBS范围
3    -5..20       15       5        24..2000
4    -5..20       15       5        24..2000
5    -5..20       15       5        24..2000
10   -5..20       15       5        24..2000
19   -5..20       15       5        24..2000
27   -5..20       15       5        24..2000
```

Sionna 加载这些 JSON 后，会插值成内部张量：

```text
phy.bler_table_interp.shape = (2, 4, 29, 85, 351)
```

这几个维度分别可以理解为：

```text
category: 0..1
table index: 1..4
MCS index: 0..28
CBS interpolation grid: 24..8424, step 100
SNR interpolation grid: -5..30 dB, step 0.1 dB
```

关键点在这里：

```text
PUSCH_table1.json 原始表只有 MCS 3..27。
但 Sionna 内部张量的 MCS 轴预留了 0..28。
没有出现在 JSON 里的 MCS，例如 0、1、2、28，会保持为 inf 占位。
```

所以之前看到 `actual_mcs=2` 时，`mcs_category=0, mcs_table_index=1, mcs_index=2` 在当前表里没有有效 BLER 曲线。Sionna 返回的是“不可用表项”，不是一个物理有效的低 MCS BLER 曲线。

### 3.2.2 下行 PDSCH table 1 长什么样

本项目下行 PDSCH、目标 BLER 10% 应使用：

```text
PDSCH_table1.json
category = 1
index = 1
```

本机文件位置：

```text
/Users/zhangwei/Downloads/lls_platform_sc_mimo/.venv-sionna1/lib/python3.10/site-packages/sionna/sys/bler_tables/PDSCH_table1.json
```

JSON 顶层说明：

```text
_comment: BLER table for PDSCH channel, Table 1
```

JSON 层级大致如下：

```json
{
  "_comment": "BLER table for PDSCH channel, Table 1",
  "category": {
    "1": {
      "index": {
        "1": {
          "MCS": {
            "3": {
              "SNR_db": [-5.0, -3.2142857142857144, "...", 20.0],
              "EbN0_db": ["..."],
              "CBS": {
                "24": {
                  "BLER": [0.9416666626930237, 0.597777783870697, "...", 0.0]
                },
                "100": {"BLER": ["..."]},
                "500": {"BLER": ["..."]},
                "1000": {"BLER": ["..."]},
                "2000": {"BLER": ["..."]}
              }
            },
            "4": {"...": "..."},
            "28": {"...": "..."}
          }
        }
      }
    }
  }
}
```

`PDSCH_table1.json` 摘要：

```text
可用 MCS: 3, 4, ..., 28
MCS 个数: 26
每个 MCS 的原始 SNR 点数: 15
原始 SNR 范围: -5.0 dB 到 20.0 dB
原始 CBS 点: 24, 100, 500, 1000, 2000
```

几个代表 MCS 的摘要：

```text
MCS  SNR范围(dB)  SNR点数  CBS点数  CBS范围
3    -5..20       15       5        24..2000
4    -5..20       15       5        24..2000
5    -5..20       15       5        24..2000
10   -5..20       15       5        24..2000
19   -5..20       15       5        24..2000
27   -5..20       15       5        24..2000
28   -5..20       15       5        24..2000
```

代表性原始 BLER 数组示例：

```text
PDSCH table 1, MCS 3, CBS 24:
SNR 前 3 个点:  -5.0, -3.2143, -1.4286 dB
BLER 前 3 个值: 0.9416667, 0.5977778, 0.1630000
SNR 后 3 个点:  16.4286, 18.2143, 20.0 dB
BLER 后 3 个值: 0.0, 0.0, 0.0

PDSCH table 1, MCS 27, CBS 2000:
SNR 前 3 个点:  -5.0, -3.2143, -1.4286 dB
BLER 前 3 个值: 1.0, 1.0, 1.0
SNR 后 3 个点:  16.4286, 18.2143, 20.0 dB
BLER 后 3 个值: 1.0, 0.2570, 0.0
```

Sionna 加载 JSON 后会插值成统一内部张量：

```text
phy.bler_table_interp.shape = (2, 4, 29, 85, 351)
```

其中：

```text
category 维: 0=PUSCH, 1=PDSCH
table index 维: 1..4
MCS index 维: 0..28
CBS 插值网格: 24..8424, step 100
SNR 插值网格: -5..30 dB, step 0.1 dB
```

对 `PDSCH_table1.json` 来说，原始 JSON 只有 MCS 3..28。因此 category=1、index=1 下的 MCS 0、1、2 仍然是不可用表项。修复后的代码不会把这些不可用项误当作 `tbler=0`。

### 3.3 修复 EESM 的数值稳定性

旧 EESM 直接计算：

```python
-beta * log(mean(exp(-sinr / beta)))
```

当真实 SINR 很高时，`exp(-sinr / beta)` 可能全部下溢成 0，导致 `log(0)`，最后写出 `effective_sinr_db=inf`。

已改为数值稳定的 log-mean-exp 写法。

修改文件：

```text
beam_sls/link.py
```

### 3.4 修复 OLLA 跨 drop 泄漏

本平台里每个 drop 是独立 UE/topology 实现。旧代码让 `olla_state` 跨 drop 保持，会让下一个 drop 的 `tti=0` 继承前一个 drop 的 offset。

已改为：

```text
OLLA 在同一个 drop 内跨 TTI 保持；
每个新 drop 开始时重置。
```

修改文件：

```text
beam_sls/sim.py
```

## 4. 实验环境

本机可用环境：

```bash
/Users/zhangwei/Downloads/lls_platform_sc_mimo/.venv-sionna1/bin/python
```

确认后端：

```text
channel backend: sionna_tr38901_uma
link adaptation backend: sionna_sys_phy_abstraction_tf
Sionna: 1.2.0
TensorFlow: 2.19.1
```

## 5. 运行记录

用户要求的默认规模命令：

```bash
env MPLCONFIGDIR=$PWD/.cache/matplotlib \
/Users/zhangwei/Downloads/lls_platform_sc_mimo/.venv-sionna1/bin/python -m beam_sls.run \
  --config configs/v2_one_site_three_sector.yaml \
  --out runs/diagnostic_drops5_tti5 \
  --num-drops 5 \
  --num-tti 5 \
  --algorithm greedy \
  --skip-heatmap
```

本机运行默认规模时，完成第 1 个 drop 后进入第 2 个 drop，随后进程退出码为 `137`。默认规模包含 30 UE、192 TX beams、24 个频点、1024 AE 阵列，本机资源压力较大。

为了验证链路逻辑，使用同样 `5 drops / 5 TTI`，但缩小 UE、波束和频点：

```bash
env MPLCONFIGDIR=$PWD/.cache/matplotlib \
/Users/zhangwei/Downloads/lls_platform_sc_mimo/.venv-sionna1/bin/python -c "
from pathlib import Path
from beam_sls.config import load_config
from beam_sls.sim import run_simulation

cfg = load_config('configs/v2_one_site_three_sector.yaml')
cfg['system']['num_drops'] = 5
cfg['system']['num_tti_per_drop'] = 5
cfg['ue_drop']['num_ut_per_sector'] = 2
cfg['measurement']['num_freq_points'] = 4
cfg['tx_array']['num_beams_h'] = 2
cfg['tx_array']['num_beams_v'] = 2
cfg['tx_array']['max_beams'] = 4
cfg['ue_array']['num_beams_h'] = 2
cfg['ue_array']['num_beams_v'] = 2
cfg['ue_array']['max_beams'] = 4
cfg['scheduler']['algorithm'] = 'greedy'
cfg['coverage_heatmap']['enabled'] = False
run_simulation(cfg, Path('runs/diagnostic_small_drops5_tti5_causal_mcs'))
"
```

最终输出目录：

```text
runs/diagnostic_small_drops5_tti5_causal_mcs
```

## 6. 最终验证结果

目标测试：

```text
targeted tests passed
```

其中包含：

- 高真实 SINR 但低预测 SINR 时，`actual_mcs` 仍按预测 SINR 选择；
- Sionna 不可用/非法 TBLER 不再被夹成 0；
- 高 SINR EESM 不再生成 `inf`。

最终 CSV 检查：

```text
link_tti.csv: 无 inf / -inf / nan
summary.csv:  无 inf / -inf / nan
```

最终 `link_tti.csv` 统计：

```text
rows: 400
nonfinite_sinr: 0
nonfinite_mcs_selection_sinr: 0
invalid_tbler: 0
min_real_sinr_db: -2.036877947294534
max_real_sinr_db: 27.977337527488654
min_mcs_selection_sinr_db: -1.3852785765712525
max_mcs_selection_sinr_db: 31.052444354393845
tbler_zero_at_real_sinr_lt_minus20: 0
ack_at_real_sinr_lt_minus20: 0
tti0_nonzero_olla_count: 0
predicted_mcs > actual_mcs: 134
predicted_mcs = actual_mcs: 246
predicted_mcs < actual_mcs: 20
groups where actual_mcs varies across TTI: 48
```

## 7. 一个典型例子

来自：

```text
runs/diagnostic_small_drops5_tti5_causal_mcs/metrics/link_tti.csv
```

```text
scheme=full_gamma, drop=0, ue_id=2, beam=29

tti  pred_mcs  actual_mcs  mcs_selection_sinr_db  real_effective_sinr_db  tbler    ack  olla_offset_db
0    19        19          11.775                 8.885                   1.00000  0    0.0
1    19        18          10.875                 8.885                   1.00000  0    0.9
2    19        17           9.975                 8.885                   0.99975  0    1.8
3    19        16           9.075                 8.885                   0.01122  1    2.7
4    19        16           9.175                 8.885                   0.01122  1    2.6
```

解释：

- 调度器预测链路质量偏高，先选 MCS 19；
- 真实等效 SINR 只有 8.885 dB，所以 MCS 19 的 TBLER 为 1，NACK；
- NACK 后 OLLA backoff 增大，`mcs_selection_sinr_db = predicted_sinr_db - backoff` 下降；
- 后续实际 MCS 逐步降到 16；
- MCS 16 在真实 SINR 8.885 dB 下 TBLER 降到约 0.011，于是 ACK。

这个结果符合“调度器只能按有限信息选 MCS，真实 SINR 只用于链路成败判定”的仿真逻辑。
