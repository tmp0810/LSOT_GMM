# 复现指南

所有命令从 `code/` 执行。输出必须指向新目录，不要覆盖 `data/` 的历史结果。

## 1. 环境分层

- 核心 API：Python 3.9+、NumPy。
- 图表：`requirements/plot.txt`，以及 TeX Live 的 standalone/TikZ/PGFPlots。
- CPU 实验：`requirements/experiments.txt`。本文件是依赖列表，不是所有平台都字节一致的 lockfile；保留的 benchmark JSON 含真实运行版本。
- 原生 HCP：C++11 编译器、Eigen3、pybind11、Python 开发头文件。Linux 常见 Eigen 路径为 `/usr/include/eigen3`；Mac 路径按安装位置指定。
- 3DGS：单独使用对应 GHAP 官方 CUDA/PyTorch/pytorch3d 环境；不要把 CPU 依赖文件当作完整 GPU 环境。

```bash
python -m pip install -e '.[plots,experiments,native]'
python tools/build_hcp.py --eigen-include /path/to/eigen3
```

编译脚本先检查官方 `base.cpp` 哈希，原文件不修改。二进制不打包，因为 Python ABI、系统和 C++ 标准库不同。
`std::nth_element` 的坐标 ties 可能随 C++ 标准库实现不同；固定编译环境是严格一致重放的条件。

## 2. 无需重跑实验的图表生成

```bash
python assets/scripts/build_assets.py runtime       # Figure 3 两图
python assets/scripts/build_assets.py illustrations # Figure 1/分组示意图
python assets/scripts/build_assets.py reduction     # Figure 4 密度重绘
python assets/scripts/build_assets.py tables        # 表源导出 + 3DGS 结果重新汇总
python assets/scripts/build_assets.py layout        # Figure 2/5 排版重放
```

`layout` 用临时目录模拟历史脚本路径，结束后只保留生成图和报告。不访问原工作区、不更改论文。
Figure 2 输入 PDF 已含曲线，因此该命令只是复现布局；Figure 5 使用保留的渲染像素，不重新渲染场景。

## 3. Figure 3 原始计时重跑

需要 Linux 原始兼容环境、CPU 52 可用、官方 DSMW 的干净 Git checkout；冻结提交与文件哈希见 `THIRD_PARTY.md`。
原始脚本主动检查 affinity，避免不知不觉改变计时协议。包内 `sliced_mw.py` 供源码核查；由于原脚本校验 Git 提交，**它本身不是完整 checkout 的替代品**。

```bash
taskset -c 52 python experiments/runtime/run.py sample \
  --official-repository /path/to/official/DSMW \
  --output generated/runtime_sample_new
taskset -c 52 python experiments/runtime/run.py dimension \
  --official-repository /path/to/official/DSMW \
  --output generated/runtime_dimension_new
```

样本量固定 `100,1000,10000,100000`；维度固定 `5,10,15,20,25,30`、`n=m=1000`。
启动器设置数值库线程为 1；不会 SSH、调度任务或代替用户选用其他机器。
高规模 MSW/MW 按原 128 MiB 单 dense matrix gate 跳过，不补值、不外推。
逐次运行顺序、预热、投影和在线边界详见 `FIGURE3_PROTOCOL_CN.md`。
历史官方 sampler 是归一化 `torch.rand`，不是球面均匀 Gaussian sampler；不能与论文新增 64/256 方向实验混称。

## 4. Counterexample coupling

```bash
python experiments/counterexample/run.py --output generated/counterexample_new
```

保留 161 点三维路径、每点实际 official HCP/NW 和 Cholesky-HMW 计价。
历史 sliced 生成函数位于 `experiments/counterexample/original_make_multi_gmm_counterexample.py`；该文件保留原样并含旧相对路径，供来源审查，不默认运行。

## 5. GMM reduction

```bash
python experiments/reduction/run_selected.py \
  --hcp-extension build/hcp/base.so --output generated/reduction_new
```

只做本地数值回归、不测 C++ 排序时间时，可显式选择：

```bash
python experiments/reduction/run_selected.py \
  --reference-order --output generated/reduction_reference_check
```

两例输入生成后必须逐元素匹配保留源数组；两方法共享初始化。
二维的 Python 排序复刻版本仅对这次已检查的设置作数值回归，不声明对所有 ties 与 C++ 字节一致。
没有 cvxopt 时仅为不使用它的 hard-ISE 分支提供“调用即报错”的导入占位，不提供虚假的优化结果。
`search_history/` 为原探索脚本，保留历史相对路径和预算；不得把它们当作预注册测试。

## 6. 3DGS/GHAP

入口与依赖在 `experiments/scene_compression/original/` 和 `project_imports/`。
这些是原服务器部署源码，保留 `/home/...`、`/data/...` 等历史位置以便核对；不要未改路径就执行 shell 脚本。

重新跑需要：

1. GHAP 官方源码冻结提交和 matching pytorch3d/renderer 环境。
2. Train 的 30000-step、Truck 的 15000-step 源检查点，以及场景图像和相机划分。
3. 将脚本中的输入、输出、上游 checkout 路径改为本机路径，并检查新输出目录。
4. 配置 `HCP_EXTENSION`；retention 0.1/0.01 用 batch 16，0.001 用 batch 2。
5. 相同输入压缩后，用共同 fine-tuning loop 和 test split 重新评估。

历史 `.pth` 是 Python pickle 格式，只加载可信的自有/官方检查点；不要加载未知来源模型。
本包不包含大型 checkpoints、原场景、CUDA 二进制，也不触发下载。
仅重新生成表格/已选细节图无需这些外部资源。
