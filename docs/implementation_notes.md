# Implementation Notes

## Gamma matrix convention

For UE `u`, service beam `m`, and potential interferer beam `n`:

$$
\Gamma[u,m,m] = \frac{S[u,m]}{N}
$$

$$
\Gamma[u,m,n] = \frac{S[u,m]}{I[u,m,n] + N},\quad m \ne n
$$

The selected RX beam is tied to the service beam `m`, not to the interferer `n`.

## Limited-feedback scheduler

The scheduler only uses the information available in the emulated UE reports.

- `baseline`: uses SU MCS/rate only.
- `topk_conflict_id`: uses SU MCS/rate minus a conflict penalty when a co-scheduled beam appears in the reported strongest-interferer set.
- `threshold_conflict_set`: same penalty rule, but the conflict set is threshold-based.
- `full_gamma`: reconstructs pairwise interference from Gamma and evaluates estimated MU SINR.

## Actual link evaluation

After scheduling, the simulator computes the realized SINR with the true channel and true co-scheduled beams. This is intentionally separated from scheduler-side estimation.

## v2 backend convention

`scenario.channel_model` selects the channel backend. For `sionna_tr38901_uma/umi/rma`, the simulator tries to instantiate the corresponding Sionna PHY TR 38.901 channel and converts CIR coefficients to `H[ue, tx_unit, freq, nrx, ntx]`. In the default shared-codebook mode, `tx_unit` identifies the TRP channel axis and is not a TXRU binding; every codeword of that TRP references the same axis. The legacy independent-polarization mode still uses a separate axis for each fixed subarray.

If `sionna.fallback_to_numpy_if_unavailable=true`, backend construction failures are recorded and the simulator falls back to the v1 geometric channel. This keeps quick runs usable on hosts where Sionna SYS/PHY optional dependencies are not fully installed.

## v2 link adaptation convention

`link_abstraction.mode=sionna_sys_precomputed_bler` tries to use Sionna SYS `PHYAbstraction` and `InnerLoopLinkAdaptation`. The MCS selection rule follows ILLA:

$$
\max\ \mathrm{MCS}\quad \mathrm{s.t.}\quad
\mathrm{TBLER}(\mathrm{MCS}, \mathrm{SINR}_{\mathrm{eff}})
\le \mathrm{target\_bler}
$$

The actual backend is written to `link_abstraction_status.json`.

## v2.1 antenna-configuration fix

The initial v2 package kept a legacy simplified TX UPA (`num_h: 8`, `num_v: 8`), which did not match the requested TRP antenna. v2.1 replaces it with explicit 3GPP-style parameters:

4 TXRUs, 1024 AEs：

$$
(M,N,P,M_g,N_g;M_p,N_p)=(16,16,2,2,1;1,1)
$$

$$
(d_H,d_V)=(0.5,0.5)
$$

`ArrayConfig.from_dict()` now accepts either the legacy format or the explicit TRP format. For the explicit format it computes:

$$
\begin{aligned}
\mathrm{num\_ae} &= M \times N \times P \times M_g \times N_g \times M_p \times N_p \\
\mathrm{num\_h} &= N \times N_g \times N_p \\
\mathrm{num\_v} &= M \times M_g \times M_p \\
\mathrm{polarization\_count} &= P
\end{aligned}
$$

The numpy fallback channel and DFT codebook use the project ordering
`[pol0 full-spatial, pol1 full-spatial, ...]`. Every channel backend retains the
complete TRP tensor `H[UE,TX_UNIT,FREQ,RX_AE,TX_AE]`, so the default TX dimension
remains 1024. With `measurement.use_panel_channel_views: true`, SLS extracts the
configured reference panel across all polarization blocks and computes with a
temporary `M*N*P=512` array. Actual-link evaluation independently extracts the
physical panel dynamically assigned to each scheduled beam. The full channel is
never overwritten or discarded.

Sionna CIR axes are consumed as
`a[B,RX,RX_AE,TX,TX_AE,PATH,TIME]` and `tau[B,RX,TX,PATH]`. The static drop model
requests one time sample and explicitly selects `TIME=0`; each configured
frequency sample is evaluated with `sum_path a*exp(-j*2*pi*f*tau)`. Sionna
`PanelArray` antenna indices are explicitly permuted into project codebook order.
An antenna-count mismatch is an error rather than a request to repeat or average
polarization dimensions.
