# 实现版本边界

## 实际包含什么

`src/hmw/` 将历史 Cholesky-HMW 路径分为：

1. `representation.py`：复制输入、严格正权重归一化，计算 `L=cholesky(Sigma)`，拼接 `(mu, vecl(L))`。
2. `hilbert.py`：加载官方 `base.cpp` 编译出的 `base.so`，递归中位数 HCP 排序，校验 permutation；没有二维投影、量化 key 或 radix 前端。
3. `transport.py`：对两序列调用官方 weighted NW，按完整特征欧氏平方代价计价，返回平方根及稀疏 coupling。

此路径只有一个 order；不包含 q=2、离线 calibration、多 order、候选图、Sinkhorn 或 spherical anchor。
参考文件为 `experiments/_original/cholesky_official_hmw.py`、`bwlinear_official_hot.py` 和 `official_hcp_backend.py`。
新封装增加有限性、对称性、原生索引范围等输入防护，未改变有效 SPD 数据的计算表达式。
需要保留历史时间口径时必须运行 `_original/` 内的脚本，不能把新增验证开销忽略。

## 不能声称已经包含的内容

当前论文 canonical 算法先移除零权重、精确合并重复 atoms、补偿累加质量，并在递归排序前按特征字典序枚举。
Figure 3 的历史实现不做这些步骤，也保留原生 NW 返回的零质量边。
因此，历史时间不能自动归于新的完整 canonical public call；任意输入重排/重复拆分下的不变性也不能从历史脚本直接推出。
本次只整理代码，没有擅自把这些操作加进历史脚本或重新标注其测量。
新 canonical 距离/检索实验的实现尚缺，需要取得后放入单独版本目录。

## 应用不是同一个评分管线

- GMM reduction：生成分组，矩匹配，固定分配的 ISE5；CTD 采用自己的重分配和支持更新。源数、目标数及 CTD 预算以具体示例为准。
- 3DGS：中心 XYZ 排序、至多 100 个代表的局部块、GHAP 几何代价的 GPU Lloyd。它不是完整 Cholesky HMW 的距离计算，也不是 ISE 重心更新。
- Figure 4 示例经过探索后选择。代码保留整个搜索的 JSON 和入口；显示的优势不应推广成所有分布均占优。

## 校验约定

- `tools/verify_sources.py` 校验所有保留原文件的 SHA-256，并检查 Python 语法。
- `tests/` 在小规模 SPD 输入上验证表示、权重、稀疏边际及与历史公式一致性。
- 提供 `HCP_EXTENSION` 时追加实际 C++ 后端的 parity 测试；缺失时明确 skip，不伪装为通过。
- `assets/scripts/build_assets.py runtime --no-tex` 从每次 raw timing 重新做 case median → 跨 case median，要求匹配论文 CSV。
- `experiments/reduction/run_selected.py` 重新生成源数组，先要求与保留 NPZ 完全一致，再计算；Python reference order 只用于非计时数值验证。
- 最终本地检查摘要见 `VALIDATION.md`。本次没有重新跑整套服务器/GPU benchmark。
