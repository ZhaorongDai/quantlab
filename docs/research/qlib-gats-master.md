# Qlib GATs and MASTER: temporal encoder + cross-stock interaction

Researched 2026-09-28 from primary sources only. Every fact cites a file and line range. "Not found" means I looked in the listed sources and it is not there.

## Sources

| Tag | Source |
|---|---|
| `gats_ts` | `microsoft/qlib@main: qlib/contrib/model/pytorch_gats_ts.py` |
| `gats` | `microsoft/qlib@main: qlib/contrib/model/pytorch_gats.py` |
| `y158` | `microsoft/qlib@main: examples/benchmarks/GATs/workflow_config_gats_Alpha158.yaml` |
| `y360` | `microsoft/qlib@main: examples/benchmarks/GATs/workflow_config_gats_Alpha360.yaml` |
| `bench` | `microsoft/qlib@main: examples/benchmarks/README.md` |
| `lstm`, `gru` | `microsoft/qlib@main: qlib/contrib/model/pytorch_lstm.py`, `pytorch_gru.py` |
| `loader` / `handler` / `ds` / `proc` | `microsoft/qlib@main: qlib/contrib/data/loader.py`, `qlib/contrib/data/handler.py`, `qlib/data/dataset/__init__.py`, `qlib/data/dataset/processor.py` |
| `paper` | MASTER, arXiv 2312.15235v1 (AAAI 2024), cited by page |
| `m_model` / `m_base` / `m_main` / `m_readme` | `SJTU-DMTai/MASTER@master: master.py`, `base_model.py`, `main.py`, `README.md` |
| `m_q` / `m_qy` | `SJTU-DMTai/MASTER@master: qlib-update/pytorch_master_ts.py`, `qlib-update/workflow_config_master_Alpha158.yaml` |
| `fork_ds` / `fork_m` / `fork_y` | `SJTU-DMTai/qlib@main` (formerly `SJTU-Quant/qlib`): `qlib/contrib/data/dataset.py`, `qlib/contrib/model/pytorch_master_ts.py`, `examples/benchmarks/MASTER/workflow_config_master_Alpha158.yaml` |
| `gat_paper` | Velickovic et al., Graph Attention Networks, arXiv 1710.10903, Eq. 1-4 (p. 3) |

**MASTER is not in `microsoft/qlib`.** A recursive listing of the `main` tree has no file matching `master` (case-insensitive), `qlib/contrib/data/dataset.py` has no `MASTERTSDatasetH` (only `MTSDatasetH`, line 102), and `bench` has no MASTER row. The qlib port lives in the authors' fork `SJTU-DMTai/qlib`. The authors call it a community port "not under the authors' maintenance" that may be inconsistent with the paper (`m_readme`, "Important Notice on 2025-06-26" and "A Qlib implementation"). The authors' own lightweight repo is `SJTU-DMTai/MASTER`, and its `qlib-update/` folder is their corrected qlib model and config.

---

## 1. GATs (qlib)

Qlib ships two variants. They share the same `GATModel` and differ in how the input is built:

- **Alpha158 variant:** `pytorch_gats_ts.GATs` with `TSDatasetH`.
- **Alpha360 variant:** `pytorch_gats.GATs` with a plain `DatasetH`.

### 1.1 Input

**Alpha158 (TS variant)**
- Dataset `TSDatasetH`, `step_len: 20` (`y158` L69-81). Handler `Alpha158` (`y158` L73-76).
- Feature subset: `FilterCol` keeps **20 Alpha158 columns**: RESI5, WVMA5, RSQR5, KLEN, RSQR10, CORR5, CORD5, CORR10, ROC60, RESI10, VSTD5, RSQR60, CORR60, WVMA60, STD5, RSQR20, CORD60, CORD10, CORR20, KLOW (`y158` L13-19). `d_feat: 20` (`y158` L57).
- Per sample: `[T=20, F=20]` plus a label column. The model takes `feature = data[:, :, 0:-1]` and `label = data[:, -1, -1]`, which is the label at the last step (`gats_ts` L201-202).
- Gaps in the time window: `dl.config(fillna_type="ffill+bfill")` on train, valid and test (`gats_ts` L244-245, L320). `TSDataSampler` forward-fills and then back-fills the *row indices* of the window, and pads the start with NaN index when fewer than `step_len` rows exist (`ds` L552-560).

**Alpha360 (non-TS variant)**
- Dataset `DatasetH` with handler `Alpha360`, and no feature filter (`y360` L62-69). `d_feat: 6` (`y360` L50).
- Alpha360 has 360 columns: 60 lags of each of close, open, high, low and vwap, all divided by the current `$close`, and volume divided by `($volume+1e-12)`. The columns are ordered lag 59 down to lag 0 within each field (`loader` L16-58).
- The model reshapes a `[N, 360]` input to `[N, 6, 60]` and then permutes it to `[N, 60, 6]`, oldest step first (`gats` L376-377).

**Feature processors (both variants):** `RobustZScoreNorm(clip_outlier=true)` and then `Fillna` on features, as infer processors (`y158` L20-26, `y360` L13-19).
- `RobustZScoreNorm` uses the median and 1.4826×MAD over `fit_start_time`..`fit_end_time` = 2008-01-01..2014-12-31, then clips to ±3 (`proc` L262-297; `y158` L9-10).
- `Fillna` defaults to 0 (`proc` L182).

### 1.2 Batching

- **TS variant:** `DailyBatchSampler` yields one index array per `datetime`, so one batch holds **all stocks of one day** and the batch size varies with the day's stock count (`gats_ts` L26-41). The DataLoader uses this sampler for train and valid, with `drop_last=True` and `num_workers=n_jobs`, default 10 (`gats_ts` L247-251). The sampler **does not shuffle**: days are visited in chronological order every epoch.
- **Non-TS variant:** `get_daily_inter` builds the same per-day slices. In training it **shuffles the order of days** (`gats` L182, and the shuffle logic in `get_daily_inter`). Evaluation does not shuffle.

### 1.3 Architecture

Source: `GATModel`, identical in both files (`gats_ts` L338-393, `gats` L326-381).

1. **Temporal encoder.** `nn.GRU` or `nn.LSTM` with `batch_first=True`, `hidden_size`, `num_layers` and inter-layer `dropout` (`gats_ts` L342-357). Take the last step: `hidden = out[:, -1, :]`, shape `[N, H]` (L387-388).
2. **Attention scores** (`cal_attention`, L371-384):
   - `x = W·h + b`, where `W` is `self.transformation = nn.Linear(H, H)` (L363, L372). The line `y = transformation(y)` is computed but never used (L373).
   - `e_x = x.expand(N, N, H)`, so `e_x[i,j] = x_j`. `e_y = e_x.transpose(0,1)`, so `e_y[i,j] = x_i`. The two are concatenated and projected onto `a ∈ R^{2H×1}`, a `torch.randn` initialised parameter (L364, L377-381).
   - The resulting score is `s_ij = LeakyReLU(a[:H]·Wh_j + a[H:]·Wh_i)`. `nn.LeakyReLU()` uses PyTorch's default negative slope of 0.01 (L368).
   - `softmax(dim=1)`, so each row `i` is normalised over all `j` (L369, L383).
   - The graph is **fully connected over every stock of the day, self included**. There is no mask and no predefined relation.
3. **Aggregation with residual.** `hidden = att_weight.mm(hidden) + hidden` (L390). Note that it aggregates the **untransformed** `h`, not `W·h`.
4. **Head.** `fc: Linear(H,H)`, then LeakyReLU, then `fc_out: Linear(H,1)`, then `.squeeze()` (L366-367, L391-393).
5. There is one attention head and one GAT layer.

**How this differs from the GAT paper.** The paper computes `α_ij = softmax_j(LeakyReLU(aᵀ[Wh_i ‖ Wh_j]))` over neighbours `N_i`, with a LeakyReLU slope of 0.2. It outputs `h'_i = σ(Σ_j α_ij W h_j)` and uses multi-head concatenation (`gat_paper` Eq. 2-5).

Qlib's version differs in five ways:
- the concatenation order is `[Wh_j ‖ Wh_i]`;
- the slope is 0.01;
- it aggregates `h` rather than `Wh`;
- it adds an explicit residual `+h`;
- it has one head and the whole cross-section as the neighbourhood.

The qlib GATs README only links the paper (`examples/benchmarks/GATs/README.md`).

**Defaults**

| Setting | Constructor default | Benchmark config |
|---|---|---|
| `d_feat` | 20 (TS) / 6 (non-TS) | 20 / 6 |
| `hidden_size` | 64 | 64 |
| `num_layers` | 2 | 2 |
| `dropout` | 0.0 | **0.7** |
| `n_epochs` | 200 | 200 |
| `lr` | 1e-3 | **1e-4** |
| `early_stop` | 20 | **10** (Alpha158) / **20** (Alpha360) |
| `metric` | `""` | `loss` |
| `base_model` | `"GRU"` | **`LSTM`** |
| `optimizer` | `adam` | `adam` |

The constructor defaults are in `gats_ts` L61-79. The benchmark values are in `y158` L56-68 and `y360` L49-61.

### 1.4 Label and preprocessing

- **Label:** `Ref($close, -2) / Ref($close, -1) - 1`, the return from t+1 close to t+2 close (`y158` L32, `y360` L25; also the handler default at `handler` L89-90 and L151-152).
- **Learn processors:** `DropnaLabel`, then `CSRankNorm` on the label (`y158` L27-31, `y360` L20-24).
- **`CSRankNorm`:** per-day percentile rank, minus 0.5, times 3.46, which gives roughly unit std (`proc` L326-358).
- **Train and valid data** are both prepared with `DK_L`, so valid labels are also rank-normed and NaN-dropped (`gats_ts` L239-240, `gats` L231-234).
- **Test data** uses `DK_I`, so labels are not processed (`gats_ts` L319). The non-TS `predict` prepares features only (`gats` L305).

### 1.5 Loss, early stopping, optimizer, pretraining

- **Loss:** MSE over entries where the label is not NaN (`gats_ts` L164-174).
- **Early-stop metric:** with `metric` in `("", "loss")` the score is `-loss` (`gats_ts` L176-182). Early stopping therefore runs on **validation MSE, not IC**. Best params are kept when `val_score > best_score`, and training stops after `early_stop` epochs without improvement (L297-306). The best state is reloaded and saved at the end (L309-310).
- **Optimizer:** Adam (or `gd` for SGD) at `lr` (L150-155), with gradient value clipping at 3.0 (L209).
- **Pretrained base model** (`gats_ts` L262-280):
  - A standalone `LSTMModel` or `GRUModel` is built. When `model_path` is set, its state dict is loaded. Every key that also exists in `GATModel` is copied in.
  - Those models define `rnn` and `fc_out: Linear(H,1)` (`lstm` L286-306, `gru` L319-339). So both the **RNN weights and `fc_out`** transfer. `transformation`, `a` and `fc` stay randomly initialised.
  - The TS variant builds the pretrain model with the config's `d_feat`, `hidden_size` and `num_layers` (L264-266). The non-TS variant builds it with **hardcoded defaults** `LSTMModel()` / `GRUModel()`, which means d_feat=6, H=64 and 2 layers (`gats` L250-252).
  - If `model_path` is None the copy still runs, but from a fresh random RNN.
- **Checkpoint paths in the benchmark configs:**
  - Alpha158: `benchmarks/LSTM/csi300_lstm_ts.pkl` (`y158` L67).
  - Alpha360: `benchmarks/LSTM/model_lstm_csi300.pkl` (`y360` L60).
  - Both files exist in the repo under `examples/benchmarks/LSTM/`, alongside the GRU equivalents. They are the outputs of the qlib LSTM benchmarks, which use `pytorch_lstm_ts` with d_feat 20 and `pytorch_lstm` with d_feat 6 (`examples/benchmarks/LSTM/workflow_config_lstm_Alpha158.yaml` L55-58, `..._Alpha360.yaml` L48-50).
  - How those `.pkl` files were produced (seed, data version) is not found.

### 1.6 Benchmark numbers

These are CSI300 results from `bench`: the mean ± std of 20 random seeds (`bench` L9).
- Split: train 2008-2014, valid 2015-2016, test 2017-01-01..2020-08-01 (`y158` L77-80).
- Strategy: `TopkDropoutStrategy` with topk 50 and n_drop 5, open cost 5 bp, close cost 15 bp, `limit_threshold` 0.095 (`y158` L33-51).

| Dataset | IC | ICIR | Rank IC | Rank ICIR | Ann. Return | IR | MDD | Cite |
|---|---|---|---|---|---|---|---|---|
| Alpha158 (20 features) | 0.0349±0.00 | 0.2511±0.01 | 0.0462±0.00 | 0.3564±0.01 | 0.0497±0.01 | 0.7338±0.19 | -0.0777±0.02 | `bench` L37 |
| Alpha360 | 0.0476±0.00 | 0.3508±0.02 | 0.0598±0.00 | 0.4604±0.01 | 0.0824±0.02 | 1.1079±0.26 | -0.0894±0.03 | `bench` L66 |

For reference, the LSTM that GATs is initialised from scores:
- Alpha158-20: IC 0.0318, AR 0.0381 (`bench` L33).
- Alpha360: IC 0.0448, AR 0.0647 (`bench` L62).

GATs has no row in the CSI500 section (`bench` L90-149).

### 1.7 Extra data

None. GATs uses only Alpha158 or Alpha360 price and volume features. It needs no industry, graph or market data, because its graph is the fully connected daily cross-section.

---

## 2. MASTER (paper + official repo; qlib port in the authors' fork)

### 2.1 Input

- **Paper settings:** lookback τ = 8 and prediction interval d = 5. Features are Alpha158. Market info has 63 features built from the CSI300, CSI500 and CSI800 indices with d′ ∈ {5, 10, 20, 30, 60} (`paper` p.5 "Datasets").
- **Per-sample tensor:** a daily batch is `(N, T=8, F=222)`: 158 factors, 63 market features and 1 label (`m_base` L96-103; `m_readme` "Form"). The label is `data[:, -1, -1]` (`m_base` L103).
- **Split:** train 2008 Q1..2020 Q1, valid 2020 Q2, test 2020 Q3..2022 Q4 (`paper` p.5). `m_qy` L75-78 encodes these as 2008-01-01..2020-03-31, 2020-04-01..06-30 and 2020-07-01..2022-12-31.
- **Market features** (`m_readme` "Market information", and the header of `data/csi_market_information.csv`):
  - For each index S′ there are 21 features: the current return `Mask($close/Ref($close,1)-1, idx)`, and then for each d in {5,10,20,30,60}:
    - `Mean(ret, d)`
    - `Std(ret, d)`
    - `Mean($amount, d)/$amount`
    - `Std($amount, d)/$amount`
  - 3 indices × 21 = 63. The CSV names `SH000300`, `SH000905` and `SH000906`, and has 63 feature columns.
  - Note that "amount mean/std" is **divided by the current day's amount**. The README pseudo-code does not show this.
  - The market features are shared by every stock on a date.
- **Qlib-fork variant** (`fork_ds` L360-430, `marketDataHandler.get_feature_config`):
  - It uses indices **sh000300, sh000903 (CSI100) and sh000905**, with **`$volume`** in place of `$amount`. It still has 63 features.
  - The authors confirm the index change, because qlib has no CSI800 (`m_readme` "A Qlib implementation").
  - `MASTERTSDatasetH` runs a separate `marketDataHandler`, joins its columns into the feature group before the label column, and builds `TSDataSampler(..., fillna_type="ffill+bfill")` (`fork_ds` L431-490).
  - Market features get their own `RobustZScoreNorm(clip_outlier)` and `Fillna` (`m_qy` L26-39).
- **Feature preprocessing:** `RobustZScoreNorm` fitted on the training span, clipped to ±3, then `Fillna(0)`. This is the same for original and open-source data (`m_readme` "Preprocessing" 1; `m_qy` L12-19).

### 2.2 Batching

- `DailyBatchSamplerRandom` yields all stocks of one `datetime` per batch, so **N varies by day** (about 300 for CSI300 and about 800 for CSI800, per `m_readme` "Form"). With `shuffle=True` it **shuffles day order** for training; test and valid do not shuffle (`m_base` L33-53, L148-151, L158, L180).
- The paper says the same: "In each batch, MASTER is jointly optimized for all u ∈ S on a particular prediction date" (`paper` p.4).

### 2.3 Architecture

Sources: `m_model` L11-202, and `m_q` L55-249, which is the same model with named submodules.

1. **Market-guided gate** (`m_model` L146-156, L195-198):
   - The gate input is `x[:, -1, 158:221]`, the market vector **at the last step only**.
   - The gate is `α = d_feat · softmax(Linear(63→158)(m) / β)`.
   - Features are rescaled as `src = x[:, :, :158] * α`, broadcast over N and T.
   - The paper gives the same formula: `α(m_τ) = F·softmax_β(W_α m_τ + b_α)`. It notes that a smaller β means stronger selection (`paper` p.3).
2. **Feature layer and positional encoding:** `Linear(158→D)`, then a fixed sinusoidal positional encoding with max_len 100 (`m_model` L11-22, L184-185).
3. **Intra-stock aggregation, `TAttention`** (`m_model` L87-143):
   - Pre-LN (`norm1`), then bias-free Q, K and V `Linear(D,D)`, then multi-head attention over **T within each stock**.
   - The softmax is **without 1/√d scaling** (L132).
   - Each head has its own attention dropout. Then `xt = x + att` (x here is the LayerNormed input), then `norm2`, then `xt + FFN(xt)`.
   - The FFN is Linear, ReLU, Dropout, Linear, Dropout.
4. **Inter-stock aggregation, `SAttention`** (`m_model` L25-84):
   - The same block, but Q, K and V are transposed to `[T, N, D]`, so attention runs **across stocks separately at each time step**.
   - It is scaled by `temperature = sqrt(D/nhead)` (L31, L73).
   - There is no stock mask, so every stock in the day attends to every stock (paper Eq. on p.4, `Z_t = FFN2(MHA2(Q2_t, K2_t, V2_t) + H2_t)`).
5. **Temporal aggregation** (`m_model` L159-170):
   - `h = W_λ z` (bias-free), with query `h[:, -1]`.
   - `λ = softmax_t(h_t · h_τ)` and `e = Σ λ_t z_t`.
   - The paper writes `λ_{u,t} ∝ exp(z_tᵀ W_λ z_τ)` (`paper` p.4). The code applies `W_λ` to every step and dots it with the transformed last step, `(W z_t)·(W z_τ)`, which differs slightly from the paper's form.
6. **Decoder:** `Linear(D→1)` (`m_model` L192).

**Defaults.** Model settings are in `m_main` L23-39, `m_q` L218-219 and L276-277, and `paper` p.5 "Implementation":

| Setting | Value |
|---|---|
| `d_feat` | 158 |
| `d_model` (D) | 256 |
| `t_nhead` (N1) | 4 |
| `s_nhead` (N2) | 2 |
| `T_dropout_rate` | 0.5 |
| `S_dropout_rate` | 0.5 |
| gate indices | 158..221 |

β depends on the source:

| Source | CSI300 | CSI800 (or other universe) | Cite |
|---|---|---|---|
| Paper | 5 | 2 | `paper` p.5 |
| `m_main` | 5 | 2 | L31-34 |
| `m_q` | 5 | 2 (any universe other than csi300) | L301-304 |
| Qlib fork | **10** | **5** | `fork_m` L281-283 |

In both qlib versions, β is **hard-overridden by `market`**, so a `beta` kwarg is ignored.

### 2.4 Label and preprocessing

- **Paper:** `r̃_u = (c_{τ+d} − c_{τ+1}) / c_{τ+1}` with d = 5, then daily Z-score normalisation (`paper` p.2).
- **Qlib config:** `Ref($close, -5) / Ref($close, -1) - 1` (`m_qy` L25; the fork has the same line). Learn processor is `DropnaLabel` only, with `CSZScoreNorm` commented out (`m_qy` L20-24).
- **The fork** instead enables `CSRankNorm` on the label (`fork_y` L22-24) and has no drop-extreme step (`m_readme` "DropExtremeLabel").
- **Training-time processing in code** (`m_base` L106-113, `m_q` L354-363):
  - `drop_extreme` sorts the day's labels and keeps indices `[int(0.025N) : -int(0.025N)]`. It drops the **top 2.5% and bottom 2.5%** of labels together with their feature rows (`m_base` L19-27), so those stocks leave that day's attention too.
  - `zscore(label)` is then applied across the day, which is CSZScoreNorm.
  - `m_q` asserts that no NaN remains (L363).
- The originally published "original" training data already had DropNA, DropExtreme and CSZScoreNorm applied. The open-source data had only DropNA (`m_readme` "Preprocessing" 2).
- **Validation and test:** all stocks are kept as model input. NaN labels are dropped only when computing loss or metrics (`m_base` L135-144, `m_readme` "About Validation").
- **Known flaw:** the published valid and test data were dumped with the *learn* processors, so they contain about 95% of stocks instead of all of them (`m_readme` "Choose a data source", "About Validation").

### 2.5 Loss, stopping, optimizer

- **Loss:** MSE on entries where the label is not NaN (`m_base` L85-88). The paper also uses MSE (`paper` p.4).
- **Optimizer:** Adam at lr 1e-5, with gradient value clipping at 3.0 (`m_base` L82, L121; `m_main` L37; `paper` p.5 "D=256, lr=10⁻⁵").
  - `m_q` defaults to lr 8e-6 (L277), and `m_qy` sets 1e-5 (L62). The fork yaml uses 8e-6.
- **Stopping:** the rule is **the training-loss threshold, not validation IC**. After each epoch, `if train_loss <= train_stop_loss_thred` (0.95), the model saves and stops (`m_base` L168-171; `m_q` L429-433). The cap is `n_epochs` 40 (`m_qy` L61).
  - The paper says "at most 40 epochs with early stopping" (p.5). The README confirms the authors ended training by the training-loss threshold (`m_readme` "About Validation").
  - The validation IC is printed but not used (`m_base` L163-165). In `m_q` the validation loop is commented out (L408, L414-426).
  - Side effect: in `m_q`, if the threshold is never reached, `best_param` is undefined at `torch.save` (L433). This was found by reading the code, not by running it.
- **Pretrained weights:** MASTER loads **no pretrained model**. Four checkpoints are provided for evaluation only (`m_readme` "Usage" 5; `m_main` L79-89).
- **Seeds:** 5 in the paper (`paper` p.5); `m_main` loops over 0-4.

### 2.6 Reported numbers

- **Qlib `microsoft/qlib` README:** not found. MASTER has no row in `bench`.
- **Paper, Table 1** (`paper` p.6):
  - Mean ± std over 5 runs.
  - Test period 2020 Q3 to 2022 Q4.
  - Portfolio: top-30 daily. AR is excess annualised return and IR is information ratio.
  - MDD is not reported.
  - The paper reports no Alpha360 variant; its features are Alpha158 only.

| Universe | IC | ICIR | RankIC | RankICIR | AR | IR |
|---|---|---|---|---|---|---|
| CSI300 | 0.064±0.006 | 0.42±0.04 | 0.076±0.005 | 0.49±0.04 | 0.27±0.05 | 2.4±0.4 |
| CSI800 | 0.052±0.006 | 0.40±0.06 | 0.066±0.007 | 0.48±0.06 | 0.28±0.02 | 2.3±0.3 |

The paper's own GAT baseline on the same setup is:
- CSI300: IC 0.054, RankIC 0.041, AR 0.19, IR 1.3.
- CSI800: IC 0.043, RankIC 0.042, AR 0.10, IR 0.7.

Ablation without the gate, "(MA)STER", on CSI300: IC 0.064, RankIC 0.074, AR 0.25, IR 2.1 (`paper` p.6 Table 2).

Results on the open-source data are said to be in `model/performance.xlsx` (`m_readme`). I did not open that file.

### 2.7 Extra data

Yes. MASTER needs **market-index daily close and amount** (or volume in the fork) for 3 broad indices: CSI300, CSI500 and CSI800 in the paper, or CSI300, CSI100 and CSI500 in the fork. These are the inputs to the 63 market-status features. Nothing else is needed beyond the stock-level Alpha158 features: no industry and no graph.

---

## 3. Side-by-side

| | GATs (qlib) | MASTER |
|---|---|---|
| Temporal encoder | 2-layer LSTM/GRU, last hidden state | Linear + PE + 1 transformer block over T, which keeps all T tokens |
| Cross-stock step | one fully connected GAT on the last hidden, with residual | multi-head self-attention across stocks **at every time step**, then temporal attention |
| Market conditioning | none | softmax gate from 63 index features, temperature β |
| Batch | one day, all stocks, variable N | the same, with shuffled day order |
| Label | 1-day return, t+1 to t+2, CSRankNorm | 4-day return, t+1 to t+5, drop 5% extremes, then daily z-score |
| Loss / stopping | masked MSE; early stop on valid MSE, patience 10/20 | masked MSE; stop when train loss ≤ 0.95, at most 40 epochs |
| Initialisation | copies `rnn` and `fc_out` from a pretrained qlib LSTM | from scratch |
