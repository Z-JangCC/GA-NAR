# 非线性几何实验分析报告

## 证据口径

本报告读取 505 条运行记录。`main` 实验只有在同一 system/model 具备至少 3 个独立 seed 时才进入“三 seed 主结论”；尺度、机制、分配和单 seed 消融均明确视为探索性证据。Bootstrap 区间基于同 seed 配对差值，样本仅 3 时区间只能描述这 3 次运行的不确定性，不能替代更大样本推断。

所有趋势均表述为关联或受控设置下的响应，不据此宣称普适因果关系。

## 1. 三 seed 主实验

- ac_power / ganr（3-seed）：NRMSE 0.002521 ± 9.582e-05，Q5/Q1 1.042 ± 0.004959，relative gap 11.4 ± 0.1607，residual 0.005913 ± 0.000447。
- ac_power / resmlp（3-seed）：NRMSE 0.003483 ± 6.481e-05，Q5/Q1 0.981 ± 0.0416，relative gap 11.69 ± 0.3251，residual 0.007388 ± 0.0005673。
- ac_power / swiglu（3-seed）：NRMSE 0.003904 ± 9.725e-05，Q5/Q1 1.029 ± 0.01391，relative gap 10.87 ± 0.04583，residual 0.00734 ± 0.0001073。
- ac_power / gelu（3-seed）：NRMSE 0.006141 ± 3.962e-05，Q5/Q1 1.025 ± 0.0007411，relative gap 11.1 ± 1.268，residual 0.009695 ± 5.043e-06。
- ac_power / silu（3-seed）：NRMSE 0.006625 ± 4.182e-05，Q5/Q1 1.007 ± 0.001552，relative gap 10.5 ± 0.4606，residual 0.008421 ± 8.932e-05。
- ac_power / relu（3-seed）：NRMSE 0.006972 ± 3.422e-05，Q5/Q1 0.9937 ± 0.001852，relative gap 12.59 ± 0.2185，residual 0.01248 ± 7.985e-05。
- ac_power / fourier（3-seed）：NRMSE 0.01181 ± 0.0005388，Q5/Q1 0.9855 ± 0.04215，relative gap 131.5 ± 7.817，residual 0.02418 ± 0.001777。
- allen_cahn / ganr（3-seed）：NRMSE 0.01185 ± 7.572e-06，Q5/Q1 1.76 ± 0.002023，relative gap 5.07 ± 0.01479，residual 0.00844 ± 3.678e-06。
- allen_cahn / resmlp（3-seed）：NRMSE 0.01217 ± 4.715e-05，Q5/Q1 1.344 ± 0.02296，relative gap 5.985 ± 0.009005，residual 0.009505 ± 5.746e-05。
- allen_cahn / swiglu（3-seed）：NRMSE 0.01227 ± 3.201e-05，Q5/Q1 1.199 ± 0.01187，relative gap 6.046 ± 0.03763，residual 0.009419 ± 1.904e-05。
- allen_cahn / gelu（3-seed）：NRMSE 0.01437 ± 3.775e-05，Q5/Q1 1.014 ± 0.005287，relative gap 5.68 ± 0.003423，residual 0.01076 ± 1.971e-05。
- allen_cahn / silu（3-seed）：NRMSE 0.01463 ± 4.956e-05，Q5/Q1 0.982 ± 0.007954，relative gap 5.662 ± 0.01267，residual 0.01075 ± 3.416e-05。
- allen_cahn / relu（3-seed）：NRMSE 0.01541 ± 2.967e-05，Q5/Q1 0.9992 ± 0.002795，relative gap 6.761 ± 0.01358，residual 0.01164 ± 2.499e-05。
- allen_cahn / fourier（3-seed）：NRMSE 0.01657 ± 8.65e-05，Q5/Q1 1.21 ± 0.01225，relative gap 49.4 ± 1.133，residual 0.0123 ± 6.965e-05。
- controlled / ganr（3-seed）：NRMSE 0.006261 ± 7.445e-05，Q5/Q1 0.9744 ± 0.03431，relative gap 0.1938 ± 0.002593，residual 0.01128 ± 0.0001445。
- controlled / resmlp（3-seed）：NRMSE 0.007573 ± 0.0005544，Q5/Q1 1.165 ± 0.04412，relative gap 0.2349 ± 0.01913，residual 0.01377 ± 0.0009038。
- controlled / swiglu（3-seed）：NRMSE 0.007814 ± 0.0003634，Q5/Q1 1.164 ± 0.031，relative gap 0.2387 ± 0.004317，residual 0.01416 ± 0.000677。
- controlled / gelu（3-seed）：NRMSE 0.009678 ± 0.0001065，Q5/Q1 1.085 ± 0.00704，relative gap 0.2383 ± 0.003144，residual 0.01803 ± 0.000263。
- controlled / silu（3-seed）：NRMSE 0.01224 ± 0.0001265，Q5/Q1 1.089 ± 0.008065，relative gap 0.2587 ± 0.002461，residual 0.02235 ± 0.000176。
- controlled / relu（3-seed）：NRMSE 0.01243 ± 7.229e-05，Q5/Q1 1.035 ± 0.007607，relative gap 0.7864 ± 0.008192，residual 0.02259 ± 0.0001301。
- controlled / fourier（3-seed）：NRMSE 0.0239 ± 0.0006747，Q5/Q1 1.109 ± 0.009167，relative gap 5.708 ± 0.05792，residual 0.04305 ± 0.001065。
- duffing / ganr（3-seed）：NRMSE 0.004337 ± 5.458e-05，Q5/Q1 0.9249 ± 0.03172，relative gap 0.4231 ± 0.008067，residual 0.007416 ± 6.681e-05。
- duffing / resmlp（3-seed）：NRMSE 0.004868 ± 0.000303，Q5/Q1 1.01 ± 0.03167，relative gap 0.4289 ± 0.02014，residual 0.008803 ± 0.0008795。
- duffing / swiglu（3-seed）：NRMSE 0.004889 ± 0.000116，Q5/Q1 1.101 ± 0.01024，relative gap 0.4142 ± 0.01588，residual 0.008584 ± 0.0002246。
- duffing / gelu（3-seed）：NRMSE 0.007991 ± 3.984e-05，Q5/Q1 1.044 ± 0.008523，relative gap 0.4406 ± 0.003085，residual 0.01438 ± 3.601e-05。
- duffing / relu（3-seed）：NRMSE 0.009487 ± 4.685e-05，Q5/Q1 1.023 ± 0.004556，relative gap 1.089 ± 0.006373，residual 0.01646 ± 5.642e-05。
- duffing / silu（3-seed）：NRMSE 0.0101 ± 9.749e-06，Q5/Q1 0.9719 ± 0.003081，relative gap 0.4497 ± 0.001966，residual 0.01934 ± 1.958e-05。
- duffing / fourier（3-seed）：NRMSE 0.02023 ± 0.0003526，Q5/Q1 0.9512 ± 0.005525，relative gap 11 ± 0.3117，residual 0.03632 ± 0.0004615。
- ieee118 / ganr（3-seed）：NRMSE 0.0005436 ± 5.325e-05，Q5/Q1 1.778 ± 0.1909，relative gap 0.65 ± 0.02205，residual 0.0006518 ± 5.168e-05。
- ieee118 / resmlp（3-seed）：NRMSE 0.007415 ± 8.713e-05，Q5/Q1 0.937 ± 0.03034，relative gap 1077 ± 2.256，residual 0.01376 ± 0.0005524。
- ieee118 / swiglu（3-seed）：NRMSE 0.007815 ± 5.804e-05，Q5/Q1 0.884 ± 0.007072，relative gap 1035 ± 5.921，residual 0.01608 ± 1.611e-05。
- ieee118 / silu（3-seed）：NRMSE 0.01006 ± 6.069e-05，Q5/Q1 1.012 ± 0.006458，relative gap 1053 ± 5.056，residual 0.01471 ± 0.0001761。
- ieee118 / gelu（3-seed）：NRMSE 0.01018 ± 8.68e-05，Q5/Q1 0.9923 ± 0.00237，relative gap 1049 ± 6.769，residual 0.01623 ± 0.0001014。
- ieee118 / relu（3-seed）：NRMSE 0.01025 ± 4.766e-05，Q5/Q1 1.021 ± 0.007735，relative gap 1148 ± 5.851，residual 0.01962 ± 0.0001012。
- ieee118 / fourier（3-seed）：NRMSE 0.01424 ± 0.0004881，Q5/Q1 1.06 ± 0.06482，relative gap 1.281e+04 ± 419.3，residual 0.02153 ± 0.001125。
- shallow_water / ganr（3-seed）：NRMSE 0.00583 ± 1.894e-05，Q5/Q1 0.9811 ± 0.001465，relative gap 7.182 ± 0.009793，residual 0.004209 ± 1.282e-05。
- shallow_water / silu（3-seed）：NRMSE 0.011 ± 0.0001242，Q5/Q1 0.8405 ± 0.00452，relative gap 7.992 ± 0.01694，residual 0.007387 ± 6.402e-05。
- shallow_water / gelu（3-seed）：NRMSE 0.01101 ± 9.901e-05，Q5/Q1 0.8742 ± 0.00442，relative gap 7.999 ± 0.007979，residual 0.007561 ± 5.147e-05。
- shallow_water / resmlp（3-seed）：NRMSE 0.01126 ± 2.936e-05，Q5/Q1 1.054 ± 0.01519，relative gap 8.253 ± 0.02146，residual 0.008087 ± 4.764e-06。
- shallow_water / swiglu（3-seed）：NRMSE 0.01209 ± 7.667e-05，Q5/Q1 1.049 ± 0.004707，relative gap 8.256 ± 0.03928，residual 0.008579 ± 3.628e-05。
- shallow_water / relu（3-seed）：NRMSE 0.01324 ± 9.214e-05，Q5/Q1 0.8966 ± 0.00699，relative gap 9.718 ± 0.01872，residual 0.009223 ± 5.358e-05。
- shallow_water / fourier（3-seed）：NRMSE 0.01816 ± 3.524e-05，Q5/Q1 1.053 ± 0.01073，relative gap 67.46 ± 1.622，residual 0.01286 ± 2.006e-05。

### GANR 与自动选择的最强非 GANR 基线（同 seed 配对）

- **ac_power**：按主实验平均 NRMSE 自动选择 `resmlp`。
  - nrmse：GANR 相对改善 27.6%，确定性 paired-bootstrap 95% CI [23.6%, 30.1%]，胜出 3/3（配对 seed=11,29,47）。
  - q5_nrmse：GANR 相对改善 28.5%，确定性 paired-bootstrap 95% CI [21.1%, 33.3%]，胜出 3/3（配对 seed=11,29,47）。
  - gap_relative：GANR 相对改善 2.46%，确定性 paired-bootstrap 95% CI [1.47%, 4.11%]，胜出 3/3（配对 seed=11,29,47）。
  - residual_mean：GANR 相对改善 19.6%，确定性 paired-bootstrap 95% CI [9.14%, 26.8%]，胜出 3/3（配对 seed=11,29,47）。
- **allen_cahn**：按主实验平均 NRMSE 自动选择 `resmlp`。
  - nrmse：GANR 相对改善 2.67%，确定性 paired-bootstrap 95% CI [2.37%, 3.01%]，胜出 3/3（配对 seed=11,29,47）。
  - q5_nrmse：GANR 相对改善 -3.5%，确定性 paired-bootstrap 95% CI [-3.8%, -3.11%]，胜出 0/3（配对 seed=11,29,47）。
  - gap_relative：GANR 相对改善 15.3%，确定性 paired-bootstrap 95% CI [15.1%, 15.4%]，胜出 3/3（配对 seed=11,29,47）。
  - residual_mean：GANR 相对改善 11.2%，确定性 paired-bootstrap 95% CI [10.6%, 11.6%]，胜出 3/3（配对 seed=11,29,47）。
- **controlled**：按主实验平均 NRMSE 自动选择 `resmlp`。
  - nrmse：GANR 相对改善 17.1%，确定性 paired-bootstrap 95% CI [11.1%, 21.6%]，胜出 3/3（配对 seed=11,29,47）。
  - q5_nrmse：GANR 相对改善 21.3%，确定性 paired-bootstrap 95% CI [11.8%, 27.9%]，胜出 3/3（配对 seed=11,29,47）。
  - gap_relative：GANR 相对改善 17.1%，确定性 paired-bootstrap 95% CI [8.5%, 23.7%]，胜出 3/3（配对 seed=11,29,47）。
  - residual_mean：GANR 相对改善 17.9%，确定性 paired-bootstrap 95% CI [12.6%, 20.7%]，胜出 3/3（配对 seed=11,29,47）。
- **duffing**：按主实验平均 NRMSE 自动选择 `resmlp`。
  - nrmse：GANR 相对改善 10.7%，确定性 paired-bootstrap 95% CI [5.47%, 15.1%]，胜出 3/3（配对 seed=11,29,47）。
  - q5_nrmse：GANR 相对改善 13.3%，确定性 paired-bootstrap 95% CI [8.66%, 16.3%]，胜出 3/3（配对 seed=11,29,47）。
  - gap_relative：GANR 相对改善 1.16%，确定性 paired-bootstrap 95% CI [-5.01%, 7.3%]，胜出 2/3（配对 seed=11,29,47）。
  - residual_mean：GANR 相对改善 15.3%，确定性 paired-bootstrap 95% CI [9.43%, 23.7%]，胜出 3/3（配对 seed=11,29,47）。
- **ieee118**：按主实验平均 NRMSE 自动选择 `resmlp`。
  - nrmse：GANR 相对改善 92.7%，确定性 paired-bootstrap 95% CI [92.2%, 93.4%]，胜出 3/3（配对 seed=11,29,47）。
  - q5_nrmse：GANR 相对改善 89.4%，确定性 paired-bootstrap 95% CI [88.3%, 91.4%]，胜出 3/3（配对 seed=11,29,47）。
  - gap_relative：GANR 相对改善 99.9%，确定性 paired-bootstrap 95% CI [99.9%, 99.9%]，胜出 3/3（配对 seed=11,29,47）。
  - residual_mean：GANR 相对改善 95.3%，确定性 paired-bootstrap 95% CI [95.1%, 95.4%]，胜出 3/3（配对 seed=11,29,47）。
- **shallow_water**：按主实验平均 NRMSE 自动选择 `silu`。
  - nrmse：GANR 相对改善 47%，确定性 paired-bootstrap 95% CI [46.5%, 47.3%]，胜出 3/3（配对 seed=11,29,47）。
  - q5_nrmse：GANR 相对改善 42.2%，确定性 paired-bootstrap 95% CI [41.8%, 42.5%]，胜出 3/3（配对 seed=11,29,47）。
  - gap_relative：GANR 相对改善 10.1%，确定性 paired-bootstrap 95% CI [9.96%, 10.3%]，胜出 3/3（配对 seed=11,29,47）。
  - residual_mean：GANR 相对改善 43%，确定性 paired-bootstrap 95% CI [42.6%, 43.2%]，胜出 3/3（配对 seed=11,29,47）。

## 2. 非线性—误差—表征缺口

- ac_power / fourier：error~κ Spearman 0.01521 ± 0.04959，gap~error Spearman 0.3666 ± 0.03651，Q5/Q1 0.9855 ± 0.04215。
- ac_power / ganr：error~κ Spearman 0.02249 ± 0.012，gap~error Spearman -0.1901 ± 0.02213，Q5/Q1 1.042 ± 0.004959。
- ac_power / gelu：error~κ Spearman 0.002093 ± 0.00353，gap~error Spearman -0.07321 ± 0.04384，Q5/Q1 1.025 ± 0.0007411。
- ac_power / relu：error~κ Spearman 0.004878 ± 0.006629，gap~error Spearman 0.02287 ± 0.01241，Q5/Q1 0.9937 ± 0.001852。
- ac_power / resmlp：error~κ Spearman -0.03262 ± 0.03736，gap~error Spearman -0.06717 ± 0.02903，Q5/Q1 0.981 ± 0.0416。
- ac_power / silu：error~κ Spearman 0.003496 ± 0.0007533，gap~error Spearman -0.09783 ± 0.01267，Q5/Q1 1.007 ± 0.001552。
- ac_power / swiglu：error~κ Spearman -0.01267 ± 0.02133，gap~error Spearman -0.0566 ± 0.03257，Q5/Q1 1.029 ± 0.01391。
- allen_cahn / fourier：error~κ Spearman 0.378 ± 0.01221，gap~error Spearman -0.006705 ± 0.02392，Q5/Q1 1.21 ± 0.01225。
- allen_cahn / ganr：error~κ Spearman 0.6843 ± 0.001443，gap~error Spearman -0.03515 ± 0.01444，Q5/Q1 1.76 ± 0.002023。
- allen_cahn / gelu：error~κ Spearman 0.08467 ± 0.005284，gap~error Spearman 0.01242 ± 0.01198，Q5/Q1 1.014 ± 0.005287。
- allen_cahn / relu：error~κ Spearman 0.087 ± 0.004，gap~error Spearman 0.02014 ± 0.0114，Q5/Q1 0.9992 ± 0.002795。
- allen_cahn / resmlp：error~κ Spearman 0.4866 ± 0.01841，gap~error Spearman -0.1211 ± 0.01521，Q5/Q1 1.344 ± 0.02296。
- allen_cahn / silu：error~κ Spearman 0.04798 ± 0.01034，gap~error Spearman 0.03147 ± 0.01526，Q5/Q1 0.982 ± 0.007954。
- allen_cahn / swiglu：error~κ Spearman 0.445 ± 0.00525，gap~error Spearman -0.1307 ± 0.03042，Q5/Q1 1.199 ± 0.01187。
- controlled / fourier：error~κ Spearman 0.09854 ± 0.01884，gap~error Spearman 0.05981 ± 0.03044，Q5/Q1 1.109 ± 0.009167。
- controlled / ganr：error~κ Spearman -0.02093 ± 0.04038，gap~error Spearman 0.04774 ± 0.03388，Q5/Q1 0.9744 ± 0.03431。
- controlled / gelu：error~κ Spearman 0.07052 ± 0.004555，gap~error Spearman 0.059 ± 0.00394，Q5/Q1 1.085 ± 0.00704。
- controlled / relu：error~κ Spearman 0.05725 ± 0.01234，gap~error Spearman 0.1017 ± 0.02476，Q5/Q1 1.035 ± 0.007607。
- controlled / resmlp：error~κ Spearman 0.1605 ± 0.01647，gap~error Spearman 0.08469 ± 0.03837，Q5/Q1 1.165 ± 0.04412。
- controlled / silu：error~κ Spearman 0.07894 ± 0.0116，gap~error Spearman 0.1081 ± 0.0066，Q5/Q1 1.089 ± 0.008065。
- controlled / swiglu：error~κ Spearman 0.1111 ± 0.00426，gap~error Spearman 0.05668 ± 0.05924，Q5/Q1 1.164 ± 0.031。
- duffing / fourier：error~κ Spearman -0.04593 ± 0.02384，gap~error Spearman 0.1905 ± 0.0702，Q5/Q1 0.9512 ± 0.005525。
- duffing / ganr：error~κ Spearman -0.07652 ± 0.02755，gap~error Spearman 0.02983 ± 0.01421，Q5/Q1 0.9249 ± 0.03172。
- duffing / gelu：error~κ Spearman 0.04944 ± 0.00876，gap~error Spearman -0.03473 ± 0.01143，Q5/Q1 1.044 ± 0.008523。
- duffing / relu：error~κ Spearman 0.02013 ± 0.006642，gap~error Spearman 0.04883 ± 0.02538，Q5/Q1 1.023 ± 0.004556。
- duffing / resmlp：error~κ Spearman 0.0003338 ± 0.02232，gap~error Spearman -0.02051 ± 0.04911，Q5/Q1 1.01 ± 0.03167。
- duffing / silu：error~κ Spearman 0.004865 ± 0.005818，gap~error Spearman 0.007244 ± 0.01861，Q5/Q1 0.9719 ± 0.003081。
- duffing / swiglu：error~κ Spearman 0.01913 ± 0.006943，gap~error Spearman 0.05806 ± 0.01098，Q5/Q1 1.101 ± 0.01024。
- ieee118 / fourier：error~κ Spearman 0.05801 ± 0.04679，gap~error Spearman 0.2029 ± 0.03984，Q5/Q1 1.06 ± 0.06482。
- ieee118 / ganr：error~κ Spearman 0.3973 ± 0.1734，gap~error Spearman -0.01101 ± 0.007504，Q5/Q1 1.778 ± 0.1909。
- ieee118 / gelu：error~κ Spearman -0.02963 ± 0.008596，gap~error Spearman -0.03632 ± 0.002027，Q5/Q1 0.9923 ± 0.00237。
- ieee118 / relu：error~κ Spearman 0.01817 ± 0.004816，gap~error Spearman -0.02838 ± 0.007258，Q5/Q1 1.021 ± 0.007735。
- ieee118 / resmlp：error~κ Spearman -0.02168 ± 0.03384，gap~error Spearman -0.117 ± 0.009399，Q5/Q1 0.937 ± 0.03034。
- ieee118 / silu：error~κ Spearman -0.042 ± 0.00617，gap~error Spearman -0.0568 ± 0.008398，Q5/Q1 1.012 ± 0.006458。
- ieee118 / swiglu：error~κ Spearman -0.0544 ± 0.008684，gap~error Spearman -0.03804 ± 0.003558，Q5/Q1 0.884 ± 0.007072。
- shallow_water / fourier：error~κ Spearman 0.1082 ± 0.01266，gap~error Spearman 0.2539 ± 0.03974，Q5/Q1 1.053 ± 0.01073。
- shallow_water / ganr：error~κ Spearman -0.04866 ± 0.004748，gap~error Spearman -0.2712 ± 0.008442，Q5/Q1 0.9811 ± 0.001465。
- shallow_water / gelu：error~κ Spearman -0.0942 ± 0.003985，gap~error Spearman -0.2233 ± 0.02609，Q5/Q1 0.8742 ± 0.00442。
- shallow_water / relu：error~κ Spearman -0.07183 ± 0.01554，gap~error Spearman -0.003232 ± 0.01824，Q5/Q1 0.8966 ± 0.00699。
- shallow_water / resmlp：error~κ Spearman 0.1171 ± 0.02762，gap~error Spearman 0.3522 ± 0.002867，Q5/Q1 1.054 ± 0.01519。
- shallow_water / silu：error~κ Spearman -0.1053 ± 0.005351，gap~error Spearman -0.252 ± 0.02408，Q5/Q1 0.8405 ± 0.00452。
- shallow_water / swiglu：error~κ Spearman 0.1159 ± 0.005545，gap~error Spearman 0.3138 ± 0.01549，Q5/Q1 1.049 ± 0.004707。
这些量共同描述高非线性区域的误差集中与 representation gap 关联；相关系数本身不建立因果方向。

## 3. Width / depth / sample 极端设置（探索性）

- width 128→1024：平均 NRMSE 0.0184→0.00884 （变化 -52%）；高端设置 Q5/Q1=1.17，高非线性惩罚仍存在。证据级别：单 seed 探索性。
- depth 2→8：平均 NRMSE 0.0162→0.0102 （变化 -36.9%）；高端设置 Q5/Q1=0.991，高非线性惩罚未观察到。证据级别：单 seed 探索性。
- samples 10000→1e+06：平均 NRMSE 0.013→0.00824 （变化 -36.4%）；高端设置 Q5/Q1=1.08，高非线性惩罚仍存在。证据级别：单 seed 探索性。

## 4. 机制因子趋势与效应量（探索性）

- alpha / ganr：level 0.1→0.49，NRMSE 极端变化 -6.34%，Spearman ρ=-0.2，标准化线性斜率=-0.389，log-ratio effect=-0.0655。
- alpha / silu：level 0.1→0.49，NRMSE 极端变化 7.71%，Spearman ρ=0.8，标准化线性斜率=0.875，log-ratio effect=0.0743。
- condition / ganr：level 1→40，NRMSE 极端变化 124%，Spearman ρ=1，标准化线性斜率=0.946，log-ratio effect=0.808。
- condition / silu：level 1→40，NRMSE 极端变化 116%，Spearman ρ=1，标准化线性斜率=0.824，log-ratio effect=0.769。
- coupling / ganr：level 0→1，NRMSE 极端变化 15.2%，Spearman ρ=0.8，标准化线性斜率=0.692，log-ratio effect=0.142。
- coupling / silu：level 0→1，NRMSE 极端变化 22.2%，Spearman ρ=0.8，标准化线性斜率=0.667，log-ratio effect=0.2。
- localization / ganr：level 0.5→2，NRMSE 极端变化 28.2%，Spearman ρ=0.8，标准化线性斜率=0.772，log-ratio effect=0.248。
- localization / silu：level 0.5→2，NRMSE 极端变化 86.7%，Spearman ρ=1，标准化线性斜率=0.992，log-ratio effect=0.624。
- rank / ganr：level 1→8，NRMSE 极端变化 23.4%，Spearman ρ=0.4，标准化线性斜率=0.905，log-ratio effect=0.211。
- rank / silu：level 1→8，NRMSE 极端变化 20.3%，Spearman ρ=1，标准化线性斜率=0.969，log-ratio effect=0.185。

### Multiplicative interaction

- SwiGLU 相对 gelu：interaction 每增加 1，NRMSE 优势缩小 12.7 个百分点；低/高端优势 39.8% / 27.1%。
- SwiGLU 相对 silu：interaction 每增加 1，NRMSE 优势缩小 8.18 个百分点；低/高端优势 55.4% / 47.3%。

## 5. 四种 capacity allocation（探索性）

- controlled / silu 的 NRMSE 排序：aligned(0.0211) < shuffled(0.0216) < uniform(0.022) < reverse(0.0223)（小者更好）。

## 6. 方向对齐关联

- 主实验 run-level：direction_alignment 与 nrmse 的 Spearman ρ=-0.0366（n=126，status=ok）。
- 主实验 run-level：direction_alignment 与 gap_relative 的 Spearman ρ=-0.85（n=126，status=ok）。
另见 `tables/direction_alignment_associations.csv` 中每个诊断文件的点级关联；run-level 与点级结果不可混为同一估计量。

## 7. GANR 配对单 seed 消融（探索性）

- ac_power / no_gate：相对同 seed Full 的 NRMSE 退化 -3.98%（0.00263→0.00252）。
- ac_power / no_geom：相对同 seed Full 的 NRMSE 退化 -19.5%（0.00263→0.00212）。
- ac_power / no_physics：相对同 seed Full 的 NRMSE 退化 93.3%（0.00263→0.00508）。
- ac_power / no_router：相对同 seed Full 的 NRMSE 退化 5.62%（0.00263→0.00278）。
- ac_power / with_router：相对同 seed Full 的 NRMSE 退化 1.27%（0.00263→0.00266）。
- allen_cahn / no_gate：相对同 seed Full 的 NRMSE 退化 0%（0.0118→0.0118）。
- allen_cahn / no_geom：相对同 seed Full 的 NRMSE 退化 0%（0.0118→0.0118）。
- allen_cahn / no_router：相对同 seed Full 的 NRMSE 退化 17.1%（0.0118→0.0139）。
- allen_cahn / with_router：相对同 seed Full 的 NRMSE 退化 -0.0054%（0.0118→0.0118）。
- controlled / no_gate：相对同 seed Full 的 NRMSE 退化 10.4%（0.00622→0.00687）。
- controlled / no_geom：相对同 seed Full 的 NRMSE 退化 13.8%（0.00622→0.00708）。
- controlled / no_router：相对同 seed Full 的 NRMSE 退化 1.5%（0.00622→0.00631）。
- controlled / with_router：相对同 seed Full 的 NRMSE 退化 0.756%（0.00622→0.00627）。
- duffing / no_gate：相对同 seed Full 的 NRMSE 退化 25.6%（0.0044→0.00552）。
- duffing / no_geom：相对同 seed Full 的 NRMSE 退化 0%（0.0044→0.0044）。
- duffing / no_router：相对同 seed Full 的 NRMSE 退化 4.96%（0.0044→0.00461）。
- duffing / with_router：相对同 seed Full 的 NRMSE 退化 0.659%（0.0044→0.00443）。
- ieee118 / no_gate：相对同 seed Full 的 NRMSE 退化 0%（0.000483→0.000483）。
- ieee118 / no_geom：相对同 seed Full 的 NRMSE 退化 348%（0.000483→0.00217）。
- ieee118 / no_router：相对同 seed Full 的 NRMSE 退化 0%（0.000483→0.000483）。
- ieee118 / with_router：相对同 seed Full 的 NRMSE 退化 0%（0.000483→0.000483）。
- shallow_water / no_gate：相对同 seed Full 的 NRMSE 退化 0%（0.00584→0.00584）。
- shallow_water / no_geom：相对同 seed Full 的 NRMSE 退化 0%（0.00584→0.00584）。
- shallow_water / no_router：相对同 seed Full 的 NRMSE 退化 0%（0.00584→0.00584）。
- shallow_water / with_router：相对同 seed Full 的 NRMSE 退化 0%（0.00584→0.00584）。

## 8. 限制与数据完整性

尺度/机制/allocation/消融若只有单 seed，只能作为生成假设和机制一致性证据；即使在受控系统中改变单一配置，也仍可能受训练随机性与有限预算影响。

自动检查记录：
- Full 与部分消融组缺少共同 seed；对应退化量标记为不可用。

## 9. GANR 适用域与失效边界

- controlled / alpha：GANR 在 4/4 个水平优于同水平最强基线；该表用于标出适用域转折，不把单因素扫描解释为普适因果。
- controlled / condition：GANR 在 4/4 个水平优于同水平最强基线；该表用于标出适用域转折，不把单因素扫描解释为普适因果。
- controlled / coupling：GANR 在 4/4 个水平优于同水平最强基线；该表用于标出适用域转折，不把单因素扫描解释为普适因果。
- controlled / interaction：GANR 在 3/3 个水平优于同水平最强基线；该表用于标出适用域转折，不把单因素扫描解释为普适因果。
- controlled / localization：GANR 在 4/4 个水平优于同水平最强基线；该表用于标出适用域转折，不把单因素扫描解释为普适因果。
- controlled / rank：GANR 在 4/4 个水平优于同水平最强基线；该表用于标出适用域转折，不把单因素扫描解释为普适因果。
