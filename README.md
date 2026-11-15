# Robustness-Lib

Reusable noise-stability checks: how much each codec degrades under Gaussian noise.

## What it does

- add_noise(matrix, std, rng) adds Gaussian noise to a copy of a matrix.
- stability_check(matrix, noise_std, repeats, rng) perturbs the matrix
  several times, runs each codec on the noisy input, and reports the
  mean relative error against the clean matrix, sorted most stable first.

    from robustness_lib import stability_check

    for row in stability_check(matrix, noise_std=0.05):
        print(row["method"], round(row["mean_relative_error"], 4))

## API

- add_noise(matrix, std=0.05, rng=None)
- stability_check(matrix, noise_std=0.05, repeats=3, rng=None)
- The four codecs are included so the checks are self-contained.

## Honest results and limitations

- Quantization is the most noise-tolerant codec here (it clips to the
  value range); Fourier least-squares fitting is the most fragile - its
  coefficients swing hard under noise. SVD sits in between.
- stable is an arbitrary 0.5 relative-error threshold, not a guarantee.
- Results depend on matrix structure; smooth matrices and random
  matrices behave very differently.

## Status

Part of the kapildhinwa2010-dev toolkit experiments. CI and CodeQL run on
every push to main. License: MIT.
