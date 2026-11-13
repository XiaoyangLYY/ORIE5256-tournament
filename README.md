# ORIE 5256 Team Project: Numerai Tournament Entry

A LightGBM model for the [Numerai](https://numer.ai) tournament, built for ORIE 5256. It reaches a **validation CORR of 0.0243** on held-out 2019–2026 data and is submitted live as Numerai model **`group_3_xl_mt`**.

## Results

Per-date scores on the validation period (2019-03 to 2026-08), after a 60-business-day embargo. Sharpe = mean / standard deviation of the per-date scores.

| Experiment | What changed | CORR mean | CORR Sharpe | Positive dates | CORR max drawdown | NCORR mean | NCORR Sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 starter | Official starter: 200k random rows, small features, 300 trees, lr 0.05 | 0.0061 | 0.48 | 68.8% | 0.280 | 0.0074 | 0.51 |
| E1 | Every 4th date, all stocks | 0.0110 | 0.78 | 77.9% | 0.317 | 0.0143 | 0.91 |
| **E2 (submitted)** | E1 + medium features, 2,000 trees, lr 0.01 | **0.0243** | **1.46** | **90.4%** | **0.115** | **0.0290** | **1.53** |
| E3 | E2 + two auxiliary targets, ranks averaged | 0.0180 | 1.21 | 88.3% | 0.181 | 0.0266 | 1.31 |

E2 was retrained on training + validation data and submitted. Numerai's own diagnostics for the submitted model (NCORR 0.171) are in-sample, because that period was used in the final training, so the held-out numbers above are the validation result.

## Method

Design choices follow López de Prado, *Advances in Financial Machine Learning* (AFML):

- **Time split with embargo (ch. 7).** Train on dates before 2019, validate on 2019–2026, and drop validation dates within 60 business days of the last training date, since the 60-day targets would otherwise leak.
- **Sampling by date (ch. 4).** Weekly targets with a 60-day horizon overlap, so we keep every 4th date with all of its stocks instead of random rows.
- **Pre-registered experiments (ch. 11–14).** E1–E3 were fixed in advance and no others were tried, to limit selection bias.
- **Model.** LightGBM regressor, `max_depth=5`, `num_leaves=31`, `colsample_bytree=0.1`.
- **Scoring.** `numerai_corr` and `neutral_correlation` from Numerai's `numerai-tools`, computed per date and averaged.

## Repository

```
notebooks/numerai_pipeline.ipynb   full pipeline: download → train E0–E3 → validation CORR/NCORR → final model → .pkl
requirements.txt                   Python packages
```

Data and model files are not committed (see `.gitignore`); the notebook downloads the data.

## How to run

1. Open `notebooks/numerai_pipeline.ipynb` in Google Colab (High-RAM runtime recommended).
2. Run all cells. Data is cached in `MyDrive/numerai_data`, so later runs skip the download. A full run takes about 2–3 hours, mostly training.
3. The last cells save `improved_numerai.pkl`, test it on the current live data, and download it.
4. Upload the `.pkl` on the model's page at numer.ai. Numerai then runs it and submits every round.

Built on Numerai's [`hello_numerai`](https://github.com/numerai/example-scripts/blob/master/numerai/hello_numerai.ipynb) example (data version v6.0).
