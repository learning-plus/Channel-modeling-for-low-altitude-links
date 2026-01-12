# Week5 Summary (Model Fitting)

## LoS Probability Model (Logistic)
p(LoS)=1/(1+a*exp(-b*(theta-c)))

| Parameter | Value |
|---|---|
| a | 10.261679 |
| b | 0.046981 |
| c | 10.210767 |
| RMSE | 0.033983 |
| MAE | 0.029954 |

## Path Loss Model (Log-distance, d0=1m)
| Condition | n | PL(d0) (dB) | sigma (dB) | RMSE (dB) | MAE (dB) |
|---|---:|---:|---:|---:|---:|
| LoS | 2.000000 | 40.045997 | 0.000000 | 0.000000 | 0.000000 |
| NLoS | 2.001887 | 60.005035 | 5.980002 | 5.980002 | 4.773319 |

## Global Stats
- samples: 31352
- los_ratio: 0.267288
- path_loss_mean/std: 98.963287/11.895045 dB
- rx_mean/std: -78.963287/11.895045 dBm

## Output Files
- fit_params.csv
- fit_metrics.json
- plos_binned.csv
- los_prob_vs_elevation.png
- path_loss_fit.png
- hist_elevation.png
- hist_distance.png
- residual_hist_los.png
- residual_hist_nlos.png

## Input Files
- D:\低空链路信道建模\output\run_big_links.npz
