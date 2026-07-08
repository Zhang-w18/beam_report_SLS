# 增益空间检查与分析报告（full_gamma vs baseline，以及 topk/threshold 逼近能力）

> 面向的问题：为什么 full_gamma 相对 baseline 的系统吞吐增益只有 ~19%（单站点）/ ~12%（三站点全局）？是否正常？为什么单站点 > 三站点？topk_conflict_id / threshold_conflict_set 为什么逼近不了 full_gamma？
>
> 本报告 = 结论 + 代码依据 + **可在服务器上直接跑、把输出粘贴回来分析的验证实验**。

---

## 0. 元说明：本报告的依据与可信度边界

**分析者没有运行任何系统级仿真。** 结论来自：(a) 逐行阅读 `beam_sls/` 全链路代码；(b) 对 `beam_sls/mcs.py` 的查表函数做算术；(c) 阅读仓库里已有的 `runs/*/metrics/` 产物；(d) 用**真实调度器代码**对合成输入做确定性单元验证（`scripts/verify_scheduler_findings.py`）。因此：

| 结论类别 | 是否依赖重仿真 | 可信度 |
|---|---|---|
| 冲突惩罚 λ=0.35 相对速率量级可忽略 | 否（代码+算术，已单元验证） | **高，代码没改就成立** |
| full_gamma 候选波束数(4) ≠ 其它(2) | 否（代码事实） | **高** |
| full_gamma 的 MU-SINR 反推正确 | 否（已单元验证，误差 <1e-6 dB） | **高** |
| MCS 在 28dB / ~1122 Mbps 饱和 | 否（查表） | **高** |
| 你的 19% / 12% 具体数字、单站 vs 三站的原因 | **是** | 需要用第 5–6 节实验在你的配置上确认 |

仓库里 `runs/diagnostic_*`、`runs/verify_*` 都是**玩具配置**（如 2 UE/扇区、2×2 波束、5 drop），只能用于展示**机制**，不能当作你服务器结果的验证。

---

## 1. 背景与仿真目标（简述）

UE 上报"服务波束 + 波束间干扰/兼容关系"，调度器据此做 MU 用户分组与服务波束选择。四种上报：

- `full_gamma`：完整 Γ 矩阵（oracle 上界）；
- `baseline`：只上报服务波束质量，不含干扰信息；
- `topk_conflict_id` / `threshold_conflict_set`：服务波束 + 强干扰源 ID（有限反馈）。

目标：让两种有限反馈**逼近 full_gamma 的增益**。增益定义（[sim.py:422-426](../beam_sls/sim.py#L422)）：
`gain = (R_scheme − R_baseline) / R_baseline`。

---

## 2. 增益是如何产生的（确认口径正确）

调度只决定"调度谁、用哪个波束"；两种方案都走**同一套真实链路评估**（[link.py:49](../beam_sls/link.py#L49) `realized_sinr_grid` → EESM → BLER → ACK → goodput），所以增益 = **调度决策质量差异**在真实信道上的体现。

- **baseline 调度器**目标函数用**单用户速率** `su_mcs`，完全不含干扰项（[scheduler.py:264-282](../beam_sls/scheduler.py#L264)）。
- **full_gamma 调度器**用上报 Γ 反推 `I = S/Γ − N`，累加所有共调度用户干扰算 MU-SINR，再选 MCS（[scheduler.py:249-263](../beam_sls/scheduler.py#L249)）。**这段经单元验证是正确的**（`verify_scheduler_findings.py` 的 CORRECTNESS 检查，预测 MU-SINR 与手算一致到 1e-6 dB）。
- **topk/threshold**：用 `su_mcs`（同 baseline）+ 一个"冲突计数 × λ"的惩罚（[scheduler.py:283-284](../beam_sls/scheduler.py#L283)）。

关键事实（已在 `runs/` 产物中确认）：**所有方案调度的用户数相同**（都填满 MU 上限）。所以增益来自"选了哪些用户/波束"，不是"选了几个"。

---

## 3. 核心发现（按对"逼近 full_gamma"目标的影响排序）

### 🔴 发现 #1：`conflict_penalty_lambda = 0.35` 比速率尺度小 2–3 个数量级，冲突惩罚形同虚设

- 代码：[scheduler.py:237](../beam_sls/scheduler.py#L237)、[scheduler.py:270-284](../beam_sls/scheduler.py#L270)。`conflict_penalty` 是**冲突对计数**（每对 +1.0），最后 `utility -= 0.35 × count`。
- 但 `utility` 单位是 **Mbps，每用户 700–1100**。0.35 ×（最多约 12 对）≈ **最多 4 Mbps**，占单用户速率 0.03–0.4%。
- **单元验证结果**（`scripts/verify_scheduler_findings.py`）：
  - λ=0.35 时惩罚只从 2714 Mbps 里扣 **2.1 Mbps**；
  - 要让 greedy 真正**避开**一个冲突对，λ 需 >≈23（与"一次冲突的速率损失"同量级）；λ=0.35/10 都不动，λ=30 才翻转。
- **产物验证结果**（`runs/diagnostic_*` 的 EXPERIMENT C）：`topk_conflict_id` 与 `threshold_conflict_set` 的调度**与 baseline 每个 drop 100% 相同**（Jaccard=1.000）。它们相对 baseline 的吞吐差异（如 −5.75% / +2.06%）是**同一调度、不同 ACK 随机采样**造成的噪声，不是信息价值。
- **结论**：这是 topk/threshold 逼近不了 full_gamma 的**首要原因**——调度器根本没在用上报的干扰信息。**这个结论不依赖你的具体配置。**

**修法（二选一，建议后者）**：(1) 把 λ 标定到"一次冲突的速率损失"量级（几十~几百）；(2) 更本质：对触发已知强干扰的候选，不用 SU 速率，而是把该用户降到一个"保守 SINR/MCS"（threshold 方案有门限，天然可映射惩罚 SINR），让目标函数真实反映冲突代价。这才是计划文档 §5.6.3 里 `Î` 惩罚的本意。

### 🟠 发现 #2：候选波束数不对称——full_gamma 有 4 个候选，其它只有 2 个

- 代码：[feedback.py:94-97](../beam_sls/feedback.py#L94)。`full_gamma` 用 `oracle_service_beam_top_k=4`，其它用 `service_beam_top_k1=2`。
- 后果：full_gamma 的增益里**混入了"波束选择自由度更大"的成分**（能用第 3/4 候选去躲干扰），topk/threshold 结构上就少这个自由度。这既部分抬高 full_gamma 的表观增益，又压低 proposed 的上限，放大差距。
- 建议：公平对比时让 proposed 与 full_gamma 用**相同 k**，单独隔离"干扰上报"的净价值。

### 🟡 发现 #3：并发规模（MU order）偏低，增益空间天然受限

- `max_mu_order` 由 RF 解析为 **4**（[rf.py:209-217](../beam_sls/rf.py#L217)：4 TXRU 子连接 × 1 TRP/扇区）。
- 站点域下 3 扇区 × 4 = 12 个 panel，却只调度 4 个用户（[rf.py:211](../beam_sls/rf.py#L211) 的 `physical_cap` 是**每扇区**量，被当作**整个站点域**的上限）。系统欠载 → 共调度用户之间干扰机会少 → **sum** 增益空间偏小。
- 注意：全局调度下这个上限是网络级的；你的三站点全局若能调度 ~12 个用户，说明 yaml 里 `max_mu_order` 已放大（这正是你说的"yaml 与本地不同"）。**需要用实验 D 确认你两个 run 的实际并发/负载是否可比。**

### ⚪ 发现 #4（次要）：OLLA 收敛暂态计入平均
[link.py:108-168](../beam_sls/link.py#L108) 对全部 TTI 求平均，baseline 初始 MCS 偏高 → 前十几个 TTI 高 BLER 暂态被计入，**略微抬高**增益。想要稳态可丢弃前 N 个 TTI。方向与"增益偏大"一致，不影响主结论。

---

## 4. 对两个核心问题的回答

### Q1：为什么 full_gamma 的 sum 增益"只有" ~19% / ~12%？正常吗？

基本正常，但**sum 这个指标低估了方法价值**。三层原因：

1. **sum 被强用户饱和稀释**。MCS 在 28dB/1122Mbps 封顶（`mcs.py` 查表）。贴着上限的强用户，干扰规避几乎不涨速率。你的三站点数据已证明重点在别处：**p05 用户吞吐 full_gamma=573 vs baseline=183 = +213%**，而 sum 只有 +12%。**真正的卖点是边缘/干扰受限用户，不是 sum。**
2. **轻载 + 强方向性**：并发 4、3 扇区朝向 30/150/270°、30GHz 窄波束 → 共调度用户本来干扰就不大 → 可规避的干扰总量有限。
3. 是否"接近物理上界"要用**实验 A（无干扰天花板）**判定：若 full_gamma 已接近天花板，则"19% 小"是合理的物理极限。

### Q2：为什么单站点(19%) > 三站点(12%)？（含对上一版解释的更正）

**更正**：之前默认三站点是 `per_site_joint`（各站独立、有未被管理的站间干扰），据此的"干扰稀释"解释**在全局调度下不成立**——全局下 Gamma 是全网稠密（[sim.py:67-80](../beam_sls/sim.py#L67) 对 global 返回 None → 测全网），full_gamma **会**把站间 pairwise 干扰也纳入目标。收回那段。

全局调度下最可能、且可检验的原因是 **greedy 次优性随问题规模放大**：

- baseline 目标可分（各用户 SU 速率相加），greedy 对它近乎最优——4 槽或 12 槽都一样；
- full_gamma 目标含用户间干扰耦合，是组合难题，greedy 是近似解，**槽越多、候选越多，离最优越远**。单站点填 4 槽（greedy≈最优 → 拿满 19%）；三站点全局要在 ~90 UE 里贪心填 ~12 槽（greedy 丢的更多 → 只剩 12%）。

可证伪：**实验 F/G**。若三站点全局改用 exhaustive（小规模）或 greedy+local-search 后增益回升接近单站点，则坐实是 greedy 次优而非物理极限。

**但要彻底钉死，需要先用实验 D 确认两个 run 除站点数外配置一致（尤其 `load%pan` 相同）。若不一致，19 vs 12 的比较本身不成立。**

---

## 5. 验证实验清单

下表按"是否需要重仿真"分组。**A–D 是纯后处理**（在你现有 run 输出上直接跑，几秒出结果，不用重跑一天）；**E–G 需要重仿真**（但 E/F 只改 yaml，不改代码）。

| 实验 | 回答什么 | 需重仿真 | 工具 |
|---|---|---|---|
| A 无干扰天花板 | 12/19% 是否接近物理上界 | 否（估计版）/ 是（精确版） | `analyze_gain.py` |
| B SINR/MCS 饱和 | sum 被饱和压缩多少 | 否 | `analyze_gain.py` |
| C 调度相似度 | topk/threshold 是否≈baseline（惩罚是否失效） | 否 | `analyze_gain.py` |
| D 配置可比性 | 单站/三站两个 run 是否同口径 | 否 | `analyze_gain.py` |
| E λ 扫描 + 统一候选 k | proposed 能否逼近 full_gamma（**核心目标**） | 是（仅改 yaml） | 见 5.E |
| F exhaustive vs greedy | greedy 次优是否是瓶颈 | 是（仅改 yaml） | 见 5.F |
| G greedy + local search | 大规模全局下 greedy 缺口 | 是（需加代码） | 见 5.G |

### 实验 A：无干扰系统容量上界（判断增益是否已接近物理极限）

**原理**：baseline 已按最大 SNR 选波束；令干扰 `I=0` 则 `SINR=SNR`，所以 **baseline-无干扰 = 可达 SU 速率之和的绝对上界 ≥ full_gamma**。
`最大可能增益 = (无干扰上界 − baseline真实) / baseline真实`。full_gamma 的实际增益若接近它 → 已近最优，"小"是合理的；若远小于它 → 有增益被留在桌上（查实验 F/G）。

**估计版（后处理，无需重仿真）**：`analyze_gain.py` 的 `[EXPERIMENT A]` 直接从 `schedules.csv` 里 baseline 各 link 的 `predicted_rate_mbps`（即 SU、无干扰速率）求和作为上界，输出 **capture ratio（achieved/max）**。

**精确版（需重仿真，可选）**：加一个开关，在真实链路评估里把干扰置零，整个 run 的 realized 就变成无干扰。补丁（3 行）：
```python
# beam_sls/link.py, realized_sinr_grid() 里 for other in links 之前加：
if cfg.get("link_abstraction", {}).get("debug_zero_interference", False):
    out[u] = sig / np.maximum(np.full_like(sig, meas.noise_power_w), 1e-30)
    continue
```
> 注意：`realized_sinr_grid` 当前签名没有 `cfg`，精确版需把 `cfg` 传进去（或读环境变量 `ZERO_INTERFERENCE=1`）。估计版已足够回答"是否接近上界"，精确版仅在估计版模棱两可时再做。

**输出解读**：`CAPTURE RATIO` ≈ 100% → full_gamma 已贴近无干扰天花板，sum 增益小是物理极限（合理）；≪ 100% → greedy 或候选限制留了增益。

### 实验 B：Effective-SINR / MCS 饱和分布（判断 sum 被饱和压缩多少）

**原理**：MCS 在 28dB 饱和。若大量 link 的 effective SINR ≥ 28dB，则它们的速率已封顶，干扰规避对 sum 无贡献 → sum 增益被压缩，应看 p05/CDF。

**工具**：`analyze_gain.py` 的 `[EXPERIMENT B]`，从 `link_tti.csv` 输出各方案 SINR 的 p05/p50/p95、`>=28dB` 比例、`MCS=28` 比例。

**输出解读**：`>=28dB` 比例高（如 >30%）→ 你的强用户饱和严重，sum 增益天然小；重点看 p05 与 SINR-CDF。

### 实验 C：调度相似度（证明惩罚是否失效 / 干扰信息是否被使用）

**原理**：若 topk/threshold 的调度与 baseline 完全相同，说明冲突惩罚没有改变任何决策（发现 #1）；full_gamma 与 baseline 差异越大，说明增益正来自重排。

**工具**：`analyze_gain.py` 的 `[EXPERIMENT C]`，逐 drop 计算各方案调度集合相对 baseline 的 Jaccard 与"完全相同占比"。

**输出解读**：topk/threshold 的 `J(ue,beam)≈1.000`、`==baseline%≈100%` → 惩罚失效（去做实验 E）；full_gamma 的 J 明显 <1 → 增益来自其重排。（在诊断 run 上已实测到 topk/threshold =1.000、full_gamma=0.867。）

### 实验 D：配置可比性核对（判断 19 vs 12 是否同口径）

**原理**：只有当"单站点"和"三站点全局"两个 run 除站点数外一致（尤其**每 panel 负载 load%pan**、`k1`、`oracle_k`、`λ`、drops）时，19% vs 12% 的对比才有意义。

**工具**：`analyze_gain.py` 的 `[CONFIG]` 与 `[CROSS-RUN COMPARISON]`，把两个 run 一起传入即可。

**输出解读**：若两 run `load%pan` 接近且 `capture%` 接近 → 12 vs 19 的差异不是 bug，是 greedy 质量/regime；若 `load%pan` 差很多 → **两 run 不同口径，先统一再比较**。

### 实验 E：λ 扫描 + 统一候选 k（**核心目标实验**，仅改 yaml，需重仿真）

**原理**：直接验证"修正惩罚标定 + 消除候选不对称后，topk/threshold 能否逼近 full_gamma"。

**怎么做**：复制你的配置，改这几项后各跑一遍（可先用小 drops 试）：
```yaml
feedback:
  service_beam_top_k1: 4          # 与 full_gamma 的 oracle_service_beam_top_k 一致，消除发现#2
  oracle_service_beam_top_k: 4
scheduler:
  conflict_penalty_lambda: 100    # 扫 [0.35(对照), 30, 100, 300]，对应发现#1
```
命令（示例，替换成你的 config/env）：
```bash
for L in 0.35 30 100 300; do
  python -m beam_sls.run --config configs/your_config.yaml \
    --out runs/lambda_$L --num-drops 10 --num-tti 50 --algorithm greedy \
    --skip-heatmap
  # 若无 --override，则先用改好的 yaml；下方 analyze 汇总
done
python scripts/analyze_gain.py runs/lambda_0.35 runs/lambda_30 runs/lambda_100 runs/lambda_300 \
  --out gain_lambda_sweep.txt
```

**输出解读**：看 `[CROSS-RUN COMPARISON]` 里 topk/threshold 的 `gain%` 随 λ 上升是否显著抬升并接近 full_gamma；以及 `[EXPERIMENT C]` 里它们相对 baseline 的 Jaccard 是否开始 <1（说明惩罚开始起作用）。**这是判断你方法能否成立的决定性实验。**

### 实验 F：exhaustive vs greedy（测 greedy 次优，仅改 yaml，需重仿真）

**原理**：在能跑 exhaustive 的小规模上（单站点、`k1/oracle_k` 小），比较 full_gamma 的 greedy 与 exhaustive 增益。若 exhaustive ≫ greedy，则"增益小/单站>三站"部分来自贪心次优。

**怎么做**：单站点配置，`scheduler.algorithm: exhaustive`（剪枝默认开），其余不变，跑同样 drops，与 greedy 结果对比。

**输出解读**：full_gamma 的 `gain%` 在 exhaustive 下明显高于 greedy → greedy 是瓶颈；两者接近 → greedy 已够好，增益小是物理极限。

### 实验 G：greedy + local search（大规模全局，需加代码，可选）

仅当实验 A 的 capture ≪ 100% 且实验 F 显示单站点 greedy 也吃亏时再做。思路（计划 §5.6.5）：对 greedy 结果做 beam-swap / user-swap 局部搜索，`Û` 上升则接受。这是三站点全局（exhaustive 跑不动）下逼近最优的可行手段。需要时我可以提供实现。

---

## 6. 一键脚本用法与要粘贴回来的内容

### 6.1 后处理分析（实验 A–D，几秒出结果，**现在就能在服务器上跑**）

```bash
# 单个 run：
python scripts/analyze_gain.py runs/你的单站点run --out gain_single.txt

# 单站点 + 三站点全局一起对比（推荐）：
python scripts/analyze_gain.py \
  runs/你的单站点run \
  runs/你的三站点全局run \
  --out gain_compare.txt
```
把 `gain_compare.txt` **整份粘贴回来**。我会从中判断：(1) 两 run 是否同口径；(2) 12/19% 离无干扰天花板多远；(3) 强用户饱和程度；(4) topk/threshold 是否≈baseline。

### 6.2 调度器机制自检（无需 Sionna，无需重仿真，秒级）

```bash
python scripts/verify_scheduler_findings.py
```
它用**真实调度器代码**证明：λ=0.35 惩罚≈2 Mbps（失效）、需 λ>~23 才翻转决策、full_gamma MU-SINR 反推正确。可直接把输出粘回来存档。

### 6.3 若要跑核心目标实验（E）
按 5.E 改 yaml 重跑后，用 `analyze_gain.py` 汇总，粘贴 `gain_lambda_sweep.txt` 回来。

---

## 7. 建议路线图（按性价比）

1. **先跑 6.1（实验 A–D）** → 立刻知道：两 run 是否可比、12/19% 是否接近上界、饱和程度、惩罚是否失效。**这一步不花仿真时间。**
2. **跑 6.2** → 归档惩罚失效与 full_gamma 正确性的确定性证据。
3. **跑实验 E（λ 扫描 + 统一 k）** → 你核心目标"逼近 full_gamma"能否成立的决定性实验。
4. 视 A 的 capture 结果决定是否做 F/G（greedy 次优）。
5. 汇报时**主推 p05 / SINR-CDF / 边缘用户增益**，sum 作辅助。

---

## 附：涉及的代码位置

| 主题 | 位置 |
|---|---|
| baseline 用 SU 速率、无干扰项 | [scheduler.py:264-282](../beam_sls/scheduler.py#L264) |
| 冲突惩罚（计数 × λ） | [scheduler.py:237](../beam_sls/scheduler.py#L237), [scheduler.py:283-284](../beam_sls/scheduler.py#L283) |
| full_gamma MU-SINR 反推（正确） | [scheduler.py:249-263](../beam_sls/scheduler.py#L249) |
| 候选波束数不对称 4 vs 2 | [feedback.py:94-97](../beam_sls/feedback.py#L94) |
| MU order 解析（每扇区口径） | [rf.py:209-217](../beam_sls/rf.py#L217) |
| 真实链路评估（含全部干扰） | [link.py:49-81](../beam_sls/link.py#L49) |
| MCS 表 / 28dB 饱和 | [mcs.py:28-45](../beam_sls/mcs.py#L28) |
| 增益/oracle_ratio 计算 | [sim.py:417-427](../beam_sls/sim.py#L417) |
| 调度域→候选波束裁剪（global 返回 None=测全网） | [sim.py:67-80](../beam_sls/sim.py#L67) |
