# 当前论文逐图逐表清单

基准：`SOURCE_MANIFEST.json` 记录的英文论文 Git 提交。
当前正文/附录共 **6 个 figure 环境，引用 10 个 PDF；9 个 table 环境**。
来源快照在 `docs/manuscript_snapshot/`；它仅用于追溯，不是另一个论文编译入口。

## 图

| 论文位置 | 内容 | 可用代码/输入 | 当前可复现范围 |
|---|---|---|---|
| Figure 1 | HMW 排序与运输示意 | `assets/tex/hmw_order_transport_schematic.tex` | 原始 TeX 可重新编译 |
| Figure 2(a) | 2D 校准后的路径反例 | `assets/layout_history/figure2a/` | 已有 PDF 的排版可重放；修订版原始数值/绘图程序未找到 |
| Figure 2(b) | 3D 多分量路径反例 | `experiments/counterexample/run.py`、`data/counterexample_historical/`、`assets/layout_history/figure2b/` | 历史 HMW coupling 可重测；保留历史 sliced 生成代码；修订版排版可重放，不能声称完整修订图从零复现 |
| Figure 3(a) | 随 n 增长的耗时 | `experiments/runtime/run.py sample`、`data/runtime_sample/`、`assets/tex/runtime_scaling_sample_hmw_server.tex` | 原始实验脚本、逐次时间、精确 CSV 汇总和 TeX 均齐全 |
| Figure 3(b) | 随 d 增长的耗时 | `experiments/runtime/run.py dimension`、`data/runtime_dimension/`、对应 TeX | 同上；注意不同于新增 canonical benchmark |
| Figure 4(a) | 四模态 128→8 密度 | `experiments/reduction/run_selected.py`、`data/reduction/planar_selected.npz`、`assets/scripts/plot_reduction.py` | 可从头生成数据、重跑两方法、重绘；当前排版提供便携重绘版，另存论文原图 |
| Figure 4(b) | 一维 100→10 密度 | 同上，`scalar_selected.npz` | 可从头重跑；与原二维设置预算不同，CTD 上限为 1000 |
| Figure 4 辅助图 | 排序连续分组示意 | `assets/tex/ordered_reduction.tex` | 原始 TeX 可重新编译 |
| Figure 5 | 3DGS 选定视角细节 | `experiments/scene_compression/`、`data/scene_compression/`、`assets/layout_history/figure5/` | 可从保留的像素输入重放排版；重新训练/渲染需要外部数据、检查点和官方 GPU 环境 |
| Figure 6 | 真实 MW/BW-edges/HMW 误差与时间 | `assets/reference/distance_cost_geometry.pdf` | **缺少原始实验脚本、逐例数据和绘图源；仅参考原图** |

`python assets/scripts/build_assets.py all` 生成上述可本地重绘的 9 个 PDF（Figure 6 除外），以及可计算的表格数值。
Figure 2/5 的输出报告明确写为“排版重放”，不是数据实验或 renderer 重新运行。

## 表

| 表 | label | 位置 | 数据生成/验证状态 |
|---|---|---|---|
| 1 | `tab:reuse` | `sections/reuse_table.tex` | 仅保留表源；检索计时原始代码/记录缺失 |
| 2 | `tab:gs-quality` | `sections/application.tex` | 可由 `data/scene_compression/` 重算展示数值 |
| 3 | `tab:local-response` | `appendices/proofs.tex` | 解析理论系数表；原始 TeX/推导保留，不是测量表 |
| 4 | `tab:timing-protocols` | `appendices/experimental_details.tex` | 文字协议表；原始 TeX 保留 |
| 5 | `tab:distance-errors` | `appendices/distance_tables.tex` | 仅表源；140 对直接距离实验代码/逐例记录缺失 |
| 6 | `tab:external-comparison` | `appendices/external_comparison.tex` | 仅表源；20 对、64/256 方向实验代码/记录缺失 |
| 7 | `tab:retrieval-tuned` | `appendices/retrieval_tuned_scores.tex` | 仅表源；原始划分、编码选择/检索运行代码及预测缺失 |
| 8 | `tab:retrieval-tuned-timing` | `appendices/retrieval_tuned_timing.tex` | 同上，缺逐次计时 |
| 9 | `tab:gs-time` | `appendices/applications.tex` | 可由压缩逐次计时、输出数和进程 wall time 重算 |

`build_assets.py tables` 输出两类文件：

- `*_reference.tex`：论文表格原文，**不是重算结果**。
- `scene_tables_recomputed.csv/json` 和 `gs_*_rows.tex`：由保留记录重新汇总的数值。

## 待补齐清单

1. 新 canonical HMW 实现及文中提到的 invariance/duplicate/zero-weight 回归测试。
2. Figure 6/Table 5 的 140 对输入、原始三轮计时、MW/HMW/BW-edges 输出、绘图脚本。
3. Table 6 的统一 canonical 比较入口、Gaussian-sphere 方向种子与输出。
4. Tables 1/7/8 的 digits 编码、19 个候选、划分索引、检索预测和计时记录。
5. Figure 2 修订版的数值和原始绘图程序（现有历史程序和 PDF 排版脚本不能替代全部来源）。
6. 重新运行 3DGS 所需上游仓库、源检查点、场景图像/相机文件、CUDA 依赖。

不能从论文四舍五入后的表格反推逐例数据；本包没有生成这种替代数据。
