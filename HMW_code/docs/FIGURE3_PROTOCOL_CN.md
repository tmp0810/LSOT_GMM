# Figure 3 实验协议总结

整理日期：2026-09-26。本文对应当前论文的运行时间比较图 `fig:runtime-scaling`，依据实际运行脚本与保留的服务器结果整理，供更新图注和实验附录使用。

最需要澄清的是：官方 MSW/DSMW 代码使用归一化的正象限随机方向，不是均匀球面采样。

## 1. 数据生成

每个实验实例生成一对具有相同分量数 $n=m$ 的配对 GMM。

### 源均值

$$
\mu_i\sim\operatorname{Unif}([-2,2]^d).
$$

### 源协方差

$$
\Sigma_i=Q_i\operatorname{diag}(\lambda_i)Q_i^\top,
$$

其中 $Q_i$ 由标准高斯随机矩阵的 QR 分解得到，并根据 $R$ 的对角线符号修正 $Q$ 的列。对数特征值先从

$$
\left[-\tfrac12\log20,\tfrac12\log20\right]
$$

独立均匀采样，再减去其均值。因此 $\det\Sigma_i=1$，源协方差的条件数不超过 $20$。

### 目标协方差

目标协方差同时扰动特征值和主方向：

$$
\Lambda_i
=R_iQ_i\operatorname{diag}
\bigl(\lambda_i\odot e^{\delta_i}\bigr)Q_i^\top R_i^\top,
$$

其中

$$
\delta_i=0.35\epsilon\,\xi_i-
\operatorname{mean}(0.35\epsilon\,\xi_i)\mathbf1,
\qquad
R_i=\exp\!\left(\frac{\pi}{6}\epsilon S_i\right),
$$

$\epsilon=0.35$，$\xi_i$ 为标准高斯向量，$S_i$ 为 Frobenius 范数归一化的随机反对称矩阵。中心化保持目标协方差的行列式为 $1$；扰动后没有再次将目标条件数截断到 $20$。

### 目标均值及均值—协方差平衡

初始目标均值为

$$
\widetilde\nu_i=\mu_i+0.30\epsilon\,\eta_i,
\qquad \eta_i\sim\mathcal N(0,I_d).
$$

随后统一缩放所有均值位移，使

$$
\frac1n\sum_i\|\mu_i-\nu_i\|^2
=
\frac1n\sum_i d_{\mathrm{BW}}^2
\bigl(\mathcal N(0,\Sigma_i),\mathcal N(0,\Lambda_i)\bigr).
$$

代码在计算缩放因子时对分子、分母使用 $10^{-12}$ 数值下限。在本实验的非退化配对扰动中，该步骤使均值与协方差的平均平方扰动量相匹配。

这一步中的 BW 计算仅用于生成数据，在计时区间外；生成器计算的矩阵平方根不会传给被比较的方法作为预计算结果。

### 权重和分量排列

源权重为

$$
a\sim\operatorname{Dirichlet}(2,\ldots,2).
$$

目标权重的实际生成规则为

$$
\widetilde b_i=\max\left\{
a_i\exp(0.45\epsilon\,\zeta_i),10^{-5}
\right\},
\qquad
b_i=\frac{\widetilde b_i}{\sum_j\widetilde b_j},
\qquad \zeta_i\sim\mathcal N(0,1).
$$

最后随机打乱目标分量的均值、协方差和权重，避免直接保留配对生成时的一一对应排列。所有方法接收同一对输入，运行前后检查数组哈希以确认输入未被修改。

实现见[数据生成器](</Users/xiangzhipeng/Code/HOT in BW space/experiments/benchmark_nc_q2_vs_official_dsmw_d5_server.py:242>)及其[辅助函数](</Users/xiangzhipeng/Code/HOT in BW space/experiments/benchmark_nc_q2_vs_official_dsmw_d5_server.py:191>)。

### 两个面板

- 分量数面板：$d=3$，$n=m\in\{10^2,10^3,10^4,10^5\}$。
- 维度面板：$n=m=1000$，$d\in\{5,10,15,20,25,30\}$。

分量数面板的数据 seed 为

$$
20261217+100003n+1009\,\texttt{case\_index},
$$

维度面板的数据 seed 为

$$
20261201+100003d+1009\,\texttt{case\_index}.
$$

## 2. 投影采样

### HMW

HMW 不采样随机投影。每个 Gaussian 直接映射为

$$
z_i=
\left(\mu_i,\operatorname{vecl}(\operatorname{chol}\Sigma_i)\right)
\in\mathbb R^{d+d(d+1)/2},
$$

再分别对两个混合调用未修改的官方 HCP 排序和 weighted NW coupling。此处没有 $q=2$ 压缩、离线 calibration、多视角或 refinement。

官方 HCP commit 为 `930ecc98d2a6c6e884d96015919fed7de7c664f0`。本路径采用递归中位数排序，没有 arctan box、位量化或整数 Hilbert key。

### MSW-16 和 DSMW-16

两者使用官方仓库 commit `5e6f32654cce84d064191ac0aaeb0d666c2b8a6f`；保留结果记录的 `sliced_mw.py` SHA256 为 `35f73b312afc47e7347edc61c57f09f5df0fa3360707200e3d3d8cfb845e7069`。

MSW 的每个方向实际为

$$
\widetilde\theta_p\sim\operatorname{Unif}(0,1)^d,
\qquad
\theta_p=\frac{\widetilde\theta_p}
{\|\widetilde\theta_p\|_2}.
$$

DSMW 每次使用一对方向

$$
\theta_p=
\frac{\widetilde\theta_p}{\|\widetilde\theta_p\|_2},
\qquad
u_p=\frac{\widetilde u_p}{\|\widetilde u_p\|_2},
$$

其中

$$
\widetilde\theta_p\sim\operatorname{Unif}(0,1)^d,
\qquad
\widetilde u_p\sim\operatorname{Unif}(0,1)^2.
$$

先将 Gaussian 投影为

$$
\left(\theta_p^\top\mu,
\sqrt{\theta_p^\top\Sigma\theta_p}\right),
$$

再沿 $u_p$ 投影到一维并计算 weighted $W_2^2$，最后平均并开方。DSMW 共使用 $16$ 对方向，不是 $16\times16$ 个组合。

因此不能写成 “directions sampled uniformly from the sphere”。准确表述是：

> normalized i.i.d. Uniform$(0,1)$ directions in the positive orthant, following the released implementation.

两种 sliced 方法均使用 $P=16$，无自适应停止。DSMW 调用 `calc_parallel_SMSW(..., pnum=16, max=False)`；MSW 调用 `calc_MSW(..., pnum=16, threshold=None)`，并由适配器提供一维 Gaussian 的平方 Wasserstein 代价函数。

每次调用前设置 `torch.manual_seed(20260817)`。设置 seed 本身在计时区间外，但方向张量的生成和归一化在计时区间内。同一方法、同一维度下，各规模、实例和重复使用相同的方向库；改变 $d$ 时，相应方向库也随之改变。Torch 默认 dtype 设为 float64，与输入一致。

官方采样代码副本见[official_sliced_mw.py](</Users/xiangzhipeng/Code/HOT in BW space/tmp/narrative_revision/official_sliced_mw.py:168>)；MSW 适配见[msw_function](</Users/xiangzhipeng/Code/HOT in BW space/experiments/benchmark_direct_svec_hcp_scaling_server.py:77>)。

## 3. 计时范围

所有方法都从原始 GMM 的权重、均值和协方差数组开始计时，使用 `time.perf_counter_ns()` 测量完整方法调用。

### HMW 包含的操作

$$
\text{复制并检查输入}
\longrightarrow\text{权重归一化}
\longrightarrow\text{两侧 Cholesky}
\longrightarrow\text{特征拼接}
\longrightarrow\text{两次官方 HCP 排序}
\longrightarrow\text{NW coupling}
\longrightarrow\text{稀疏边计价}
\longrightarrow\sqrt{\cdot}.
$$

稀疏边按完整 Cholesky 特征的平方欧氏代价计价。输入有效性、排序排列合法性和 coupling 边际检查也在调用范围内。

本路径每次重新计算两侧 Cholesky 表示；没有预计算复用，也没有 duplicate merging 或额外 canonical enumeration。因此这些数值应标为当前官方 HCP 路径的在线时间，不能直接等同于包含额外规范化步骤的其他完整 API 计时。

实现见[HMW 完整调用](</Users/xiangzhipeng/Code/HOT in BW space/experiments/cholesky_official_hmw.py:25>)。

### DSMW 包含的操作

NumPy-to-Torch 适配、方向生成及归一化、两层投影、投影方差开方、16 次一维 OT、平均、返回值验证及最终开方。

### MSW 包含的操作

NumPy-to-Torch 适配、16 个方向生成及归一化、投影与投影方差开方、所有 $n\times m$ 投影 Gaussian 代价、平均代价矩阵、外层精确 EMD、返回值验证及最终开方。

### Exact MW 包含的操作

源协方差主平方根、完整 $n\times m$ Gaussian BW ground matrix 构造及其逐对矩阵运算、POT 精确 EMD、目标值验证及最终开方。计时不复用数据生成器计算的平方根。

### 不计入的操作

- synthetic data generation；
- import、编译和 warm-up；
- 设置投影随机 seed；
- 输入哈希和 mutation 检查；
- 方法之外的评估、结果保存、汇总统计及绘图。

warm-up 使用 $n=m=32$ 的小实例；维度面板在每个所测维度上分别 warm-up。

### 运行环境

- Dell PowerEdge T640；
- Python 3.9.25、NumPy 2.0.2、CPU PyTorch 2.8.0；
- CPU affinity 固定为逻辑 CPU 52；
- OpenMP、OpenBLAS、MKL、VECLIB、NUMEXPR、NUMBA、BLIS 和 Torch 线程数设为 1；
- 计时循环中关闭 Python GC；
- 方法调用顺序按实例及重复编号循环轮换。

## 4. 重复次数与统计

### 分量数面板

| $n=m$ | 数据实例数 | HMW/DSMW 每实例重复次数 | MSW/MW 每实例重复次数 |
|---:|---:|---:|---:|
| $10^2$ | 5 | 31 | 7 |
| $10^3$ | 3 | 9 | 3 |
| $10^4$ | 1 | 5 | 未运行 |
| $10^5$ | 1 | 3 | 未运行 |

当 $n\ge10^4$ 时，一个 $n\times n$ float64 矩阵已经超过预设的 $128$ MiB gate，因此 MSW 和 Exact MW 不运行，也不做外推。这里的 $128$ MiB 是单个矩阵的准入阈值，不是整个进程的峰值内存测量。

### 维度面板

- 每个维度生成 3 个实例；
- HMW、DSMW 各重复 7 次；
- MSW 重复 3 次；
- Exact MW 每个实例运行一次，并只测 $d=5$。

Exact MW 在该面板没有实例内的重复计时，因此不能从这些记录估计它的实例内时间波动。

### 汇总规则

每个实例先取重复运行的中位数，图中再取所有实例中位数的中位数：

$$
t_{\mathrm{plot}}
=\operatorname{median}_{c}
\left[\operatorname{median}_{r}t_{c,r}\right].
$$

图中时间单位为秒；原始 JSON 中的时间单位为毫秒。

## 5. 可直接放进英文附录的精简版本

Figure 3 uses paired synthetic GMMs with equal source and target cardinalities. Source means are sampled uniformly from $[-2,2]^d$. Source covariance eigenbases are obtained by QR factorization of standard Gaussian matrices with diagonal-sign correction, while centered log-eigenvalues are sampled such that the determinant is one and the condition number is at most 20. Target components are obtained by perturbing the log-eigenvalues, rotating the eigenbases, and perturbing the weights, using a perturbation parameter of $0.35$. Mean displacements are rescaled so that their average squared magnitude equals the average paired covariance Bures--Wasserstein cost. Target components are randomly permuted before evaluation. All methods receive the same raw GMM pair in each case.

HMW includes input copying and validation, weight normalization, Cholesky factorization of both input mixtures, construction of the $d+d(d+1)/2$-dimensional chart, two calls to the unmodified official HCP ordering, weighted NW matching, sparse-edge pricing in the full Cholesky chart, and the final square root. This timing path does not perform duplicate merging or additional canonical enumeration. MSW and DSMW use $P=16$ directions generated by the released implementation: i.i.d. Uniform$(0,1)$ vectors normalized to unit length, supported in the positive orthant rather than sampled uniformly from the full sphere. DSMW uses 16 paired primary and secondary directions. Projection generation and normalization, tensor conversion, projection, transport computation, and final aggregation are included in the sliced-method timings. Exact MW includes construction of the full Gaussian BW cost matrix and the exact EMD solve. The projection seed is reset to `20260817` before each sliced-method call, outside the timed interval.

Data generation, imports, compilation, warm-up, input-mutation checks, serialization, and plotting are excluded. All runs use one pinned CPU core and one numerical-library thread on a Dell PowerEdge T640 server, with Python 3.9.25, NumPy 2.0.2, and CPU PyTorch 2.8.0. Method order is rotated across repetitions, and garbage collection is disabled during the timing loops. Points report the median across case-level timing medians; no missing dense-method results are extrapolated.

For the component-count panel, $d=3$ and $n=m\in\{100,1000,10^4,10^5\}$, with respectively $5,3,1,1$ independently generated cases. HMW and DSMW use respectively $31,9,5,3$ repetitions per case. MSW and exact MW use seven repetitions at $n=100$ and three at $n=1000$; they are omitted at larger sizes under a preset 128 MiB single-dense-matrix limit. For the dimension panel, $n=m=1000$ and $d\in\{5,10,15,20,25,30\}$, with three cases per dimension. HMW and DSMW use seven repetitions, MSW three, and exact MW one per case. Exact MW is evaluated only at $d=5$ under the preset dimension limit.

## 6. 对应代码与记录

- [分量数计时入口](</Users/xiangzhipeng/Code/HOT in BW space/experiments/benchmark_hmw_sample_scaling_server.py>)。
- [维度计时入口](</Users/xiangzhipeng/Code/HOT in BW space/experiments/benchmark_hmw_dimension_scaling_server.py>)。
- [HMW 实现](</Users/xiangzhipeng/Code/HOT in BW space/experiments/cholesky_official_hmw.py>)。
- [官方 HCP 适配器](</Users/xiangzhipeng/Code/HOT in BW space/experiments/official_hcp_backend.py>)。
- [分量数结果 JSON](</Users/xiangzhipeng/Code/HOT in BW space/experiments/results/HMW_SAMPLE_DECADES_SERVER_20260917_01/benchmark.json>)。
- [维度结果 JSON](</Users/xiangzhipeng/Code/HOT in BW space/experiments/results/HMW_DIMENSION_SCALING_SERVER_20260917_01/benchmark.json>)。
- [分量数绘图 CSV](</Users/xiangzhipeng/Code/HOT in BW space/paper/figures/runtime_scaling_sample_hmw_server.csv>)。
- [维度绘图 CSV](</Users/xiangzhipeng/Code/HOT in BW space/paper/figures/runtime_scaling_dimension_hmw_server.csv>)。
