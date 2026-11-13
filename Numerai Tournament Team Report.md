# Numerai Tournament Team Report

Nov 12, 2026

## Summary

E2, a LightGBM regressor using Numerai v6.0 medium features, achieved a mean CORR of **0.0243** and a mean NCORR of **0.0290** on the 2019–2026 model-selection validation period. Its CORR Sharpe was 1.46, with positive CORR on 90.4% of dates. We then retrained E2 on sampled dates from both the training and validation sets and deployed it to Numerai Classic as **`group_3_xl_mt`**. The upload log recorded 7,113 live predictions. Live scores were pending when this report was written.

- **E1–E3 experiment code and final retraining** — `hello_numerai.ipynb`, which selects the experiment by mean validation CORR and exports `improved_numerai.pkl`. The report records E2 as the selected experiment. The experiment notebook contains code without saved execution results.

## Data

We used Numerai's v6.0 dataset, downloaded with the official `numerapi` client and cached in a shared Google Drive folder so both team members work from one copy.

- **Rows:** one stock (`numerai_ticker`) on one weekly `date`. The validation file alone has about 2.57 million scored rows from 2019 to 2026.
- **Features:** 1,040 anonymised features, each binned into five integer levels (0–4). We used the official `small` (100 numeric) and `medium` (350 numeric) sets and dropped the text feature `feature_country`.
- **Target:** `target` (alias of `target_jupiter_60`), a stock-specific return over the next 60 business days, binned into 0, 0.25, 0.5, 0.75 and 1. The file lists 24 targets in total.
- **Files:** `train.parquet` (dates before 2019), `validation.parquet` (2019–2026) and `validation_neutralizer.parquet` (the risk exposures used for NCORR).

## Methodology

We used a chronological split inspired by López de Prado’s *Advances in Financial Machine Learning* (AFML). The candidate models were fitted on earlier dates and compared on a later validation period.

1. **Time split with an embargo (AFML ch. 7).** We train only on `train.parquet` and score only on `validation.parquet`, never shuffling across time. Each target looks 60 business days ahead, so the last training labels overlap the first validation weeks. We therefore drop every validation date within 60 business days of the last training date.
2. **Sampling by date, not by row (AFML ch. 4).** The 60-business-day return windows of neighbouring weekly dates overlap. We retain every fourth training date and keep its stock cross-section, subject to target availability for each fit. This reduces the number of dates loaded relative to using all training dates. E0 uses 200,000 randomly sampled rows.
3. **Model.** LightGBM gradient-boosted trees (`max_depth=5`, `num_leaves=31`, `colsample_bytree=0.1`). LightGBM samples 10% of the available features for each tree.
4. **Scoring per date.** We compute scores separately for each validation date using `numerai-tools` 0.7.2. CORR uses the signed power transformation below. NCORR ranks and Gaussianizes the predictions, removes exposure to Numerai’s neutralizer matrix, and computes Pearson correlation with the target without the signed power transformation. We summarize the weekly validation cross-sections by mean, standard deviation, unannualized Sharpe (mean divided by standard deviation), share of positive dates, and maximum drawdown of cumulative scores.

For each date, we convert average prediction ranks to percentiles using (rank − 0.5) / number of stocks, then Gaussianize them. CORR is the Pearson correlation after applying a signed 1.5 power to those values and the centered targets.

```latex
\mathrm{CORR}_t = \rho\left( \operatorname{sign}(g_t)\,|g_t|^{1.5},\ \operatorname{sign}(y_t - \bar y_t)\,|y_t - \bar y_t|^{1.5} \right), \quad g_t = \Phi^{-1}\!\left(\frac{\operatorname{rank}_{\mathrm{avg}}(\hat y_t)-0.5}{n_t}\right)
```

## Experiments and validation results

E2 ranked first on all metrics shown. Its mean CORR was 0.0243, compared with the reported E0 reference of 0.0061, and its maximum cumulative CORR drawdown was 0.115. The E0 comparison is provisional until the validation dates, stock coverage, target and scoring implementation are verified to match. We specified E1–E3 before running them and selected E2 by mean validation CORR.

| Experiment | What changed | CORR mean | CORR Sharpe | Positive dates | CORR max drawdown | NCORR mean | NCORR Sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 starter | Official notebook: small features, 200k random rows, 300 trees, learning rate 0.05 | 0.0061 | 0.48 | 68.8% | 0.280 | 0.0074 | 0.51 |
| E1 | E0 with every 4th date, all stocks | 0.0110 | 0.78 | 77.9% | 0.317 | 0.0143 | 0.91 |
| **E2** | E1 with medium features, 2,000 trees, learning rate 0.01 | **0.0243** | **1.46** | **90.4%** | **0.115** | **0.0290** | **1.53** |
| E3 | E2 plus models on two 20-day auxiliary targets, ranks averaged | 0.0180 | 1.21 | 88.3% | 0.181 | 0.0266 | 1.31 |

E1–E3 were compared on the model-selection validation period (March 2019 to August 2026), after the 60-business-day gap. The mean columns average the per-date scores. The remaining columns summarize their variability, sign or cumulative drawdown.

- **Sampling by date (E0 → E1)** increased mean CORR from 0.0061 to 0.0110. Both the sampling scheme and the number of training rows changed.
- **Capacity (E1 → E2)** produced the largest increase in mean CORR. E2 jointly expanded the feature set, increased the tree count and lowered the learning rate.
- **Auxiliary targets (E2 → E3)** lowered both mean CORR and mean NCORR. The notebook selects `target_raw_return_20` and `target_factor_neutral_20` with the available v6.0 metadata. The shorter target horizon is one possible explanation, and the experiment does not isolate its effect.

<img width="768" height="262" alt="image" src="https://github.com/user-attachments/assets/bcc24005-71f5-45d3-bf95-9df5e1e9f6df" />

## Final model and live submission

We submitted E2, retrained on training plus validation data, as Numerai model **group\_3\_xl\_mt**. The retraining step incorporated more recent observations after model selection.

1. Sampled every fourth date separately from the training set and from the eligible validation period, then combined the samples for E2 retraining. The reported run used 306 dates.
2. Packaged the model and a `predict(live_features, live_benchmark_models)` function with `cloudpickle` into a `.pkl` file.
3. Tested the file in Colab on the current `live.parquet` before uploading.
4. Uploaded it to Numerai. Numerai now loads it every round, predicts on that round's live features and submits the ranked predictions automatically.

Numerai's first run (Python 3.13) loaded the 2026-11-13 live data, generated 7,113 predictions and uploaded them successfully (HTTP 204).

<img width="724" height="233" alt="image" src="https://github.com/user-attachments/assets/f4ac40e1-42eb-49b9-bba7-de6824947e4c" />


## Discussion

Numerai's diagnostics page scores the submitted model at NCORR 0.1711 (Sharpe 7.03). The diagnostic period overlaps the final model’s training data, so the diagnostic score cannot estimate out-of-sample performance.

| Score of the same E2 setup | Model-selection validation (candidate E2) | Numerai diagnostics (retrained E2) |
| --- | --- | --- |
| NCORR mean | 0.0290 | 0.1711 |
| NCORR Sharpe | 1.53 | 7.03 |

The final fit used every fourth eligible validation date. The 60-business-day return windows on nearby dates can overlap those used for training, so the full diagnostic period provides no independent test. The CORR of 0.0243 refers to candidate E2 before retraining.

<img width="767" height="281" alt="image" src="https://github.com/user-attachments/assets/552041e1-eb8e-4687-8be7-1088e9aa324e" />


**Limitations**

- One validation period. We did not run purged k-fold cross-validation on the training data, so our E1–E3 choice rests on a single 2019–2026 split.
- Only E0 was evaluated against the fixed public `lgbm_jupiter_60` benchmark (mean contribution −0.0033). E2’s benchmark contribution remains unmeasured. Classic v6 payout scoring uses NCORR and NMMC, while our experiment selected on offline CORR and also reported NCORR.
- Live results take about three months to resolve and were not available for this report.

## Conclusion

E2 had the highest mean CORR (0.0243) and NCORR (0.0290) on the model-selection validation period. The combined change to medium features, tree count and learning rate produced the largest CORR increase among the experiments. The tested auxiliary-target ensemble scored lower. We deployed a retrained E2 as `group_3_xl_mt`. An independent evaluation would require later dates excluded from both model selection and retraining, with a gap to account for the overlapping target windows.

## References

- [Numerai CORR definition](https://docs.numer.ai/numerai-tournament/scoring/correlation-corr)
- [Numerai Classic scoring and diagnostics](https://docs.numer.ai/numerai-tournament/scoring)
- Marcos López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018.
