# Fit Results

## LoS Probability Model
p(LoS)=1/(1+a*exp(-b*(theta-c)))

- a: 10.261679
- b: 0.046981
- c: 10.210767
- RMSE: 0.033983
- MAE: 0.029954

## Path Loss Model (PL(d)=PL(d0)+10n*log10(d/d0), d0=1m)
- LoS: n=2.000000, PL(d0)=40.045997 dB, sigma=0.000000 dB
  RMSE=0.000000 dB, MAE=0.000000 dB
- NLoS: n=2.001887, PL(d0)=60.005035 dB, sigma=5.980002 dB

  RMSE=5.980002 dB, MAE=4.773319 dB

## Global Stats
- samples: 31352
- los_ratio: 0.267288
- path_loss_mean/std: 98.963287/11.895045 dB
- rx_mean/std: -78.963287/11.895045 dBm

## Inputs
- D:\低空链路信道建模\output\run_big_links.npz
