# Week6 Summary (Baselines Comparison)

## Assumptions
- 3GPP UMa: BS=ground (h_bs=25m), UT=UAV, height_handling=clamp
- UMa LoS probability uses d2D (Fig.A1 gives the reference curve)
- UMi-SC enabled: False
- Al-Hourani Urban/Dense used as baseline A2G models

- Residual CDF uses absolute error |PL_obs - PL_baseline|

## Data Quality
- total: 31352
- bad_total: 0
- bad_distance: 0
- bad_elevation: 0
- bad_path_loss: 0
- bad_rx: 0
- bad_outlier: 0

## Output Files
- plos_vs_elevation_with_baselines.png
- path_loss_vs_logd_with_baselines.png
- residual_cdf_pathloss_baselines.png
- plos_vs_d2d_3gpp_reference.png
- table_metrics_plos.csv
- table_metrics_pathloss.csv
- baseline_config.json
- week6_metrics.json

## Input Files
- D:\低空链路信道建模\output\run_big_links.npz
