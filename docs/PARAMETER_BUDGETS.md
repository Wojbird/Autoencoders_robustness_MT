# Parameter budgets

Counts below include every trainable parameter in the autoencoder/generator. For
VQ-VAE they include the 512-entry codebook. The adversarial discriminator is not
included in the generator count and has **185,889** trainable parameters.

| Latent channels | Conv transpose | Residual | U-shaped, no skips | U-Net | Adversarial U-Net generator | VQ-VAE | Max spread |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 292,595 | 291,467 | 291,995 | 299,671 | 299,671 | 289,143 | 3.64% |
| 32 | 309,407 | 315,915 | 305,783 | 301,303 | 301,303 | 308,229 | 4.85% |
| 64 | 336,119 | 338,923 | 339,703 | 325,383 | 325,383 | 330,405 | 4.40% |
| 128 | 546,431 | 547,567 | 538,683 | 556,963 | 556,963 | 544,737 | 3.39% |
| 256 | 839,063 | 845,855 | 845,559 | 839,323 | 839,323 | 841,741 | 0.81% |

`U-shaped` and `U-Net` are matched-capacity architecture comparisons, not a strict
single-variable skip ablation: their channel widths differ to keep total parameter
budgets close. A strict skip-only ablation would use identical widths and explicitly
accept the additional fusion parameters in U-Net.

The table was calculated from the declared layers. Re-run a framework-level count
after installing PyTorch and before starting the full training campaign; the future
`param_counter.py` update should treat any discrepancy as an error.
