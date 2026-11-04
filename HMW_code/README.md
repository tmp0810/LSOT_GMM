# HMW 论文代码与图表复现包

本目录是按 **2026-09-26 当前英文论文**整理的本地仓库，不是旧 slides 的代码集合。
它包含已找到的核心实现、原始实验入口、绘图代码、测量记录、第三方源码及来源清单。
原项目和 `paper/` 均未移动或修改；没有创建远程仓库或推送。

**覆盖边界：这不是已完全复现的整篇论文发布版。** 新增 Figure 6、检索/直接距离实验的原始运行代码与逐例结果尚未在本地找到；这些资产保留论文原图和表格定义并显式标为缺项。不要把参考 PDF 的存在理解为该实验可以重跑。

## 目录

```text
code/
├── src/hmw/                  # 简洁 API：Cholesky 表示 → 官方 HCP → NW/计价
├── experiments/
│   ├── runtime/              # Figure 3 样本量、维度计时入口
│   ├── counterexample/       # 3D 路径上真实 HMW coupling 验证
│   ├── reduction/            # 两个选定 GMM reduction 示例及搜索历史
│   ├── scene_compression/    # 3DGS/GHAP 集成、原始服务器脚本
│   └── _original/            # 不改算法的原始脚本及本地 import 依赖
├── assets/
│   ├── scripts/              # 统一重绘/汇总入口
│   ├── tex/                  # 示意图、运行时间图的原始 TeX
│   ├── reference/            # 论文实际引用的 10 个 PDF，仅作对照
│   └── layout_history/       # Figure 2/5 已保留的排版脚本和输入
├── data/                     # 原始计时 JSON、选定 NPZ、3DGS 指标记录
├── third_party/              # HCP、CTDGMR、DSMW 源码快照
├── tests/                    # 小规模单元测试/可选原生后端一致性测试
├── tools/                    # HCP 编译、原始文件校验、打包
├── requirements/            # 绘图与实验依赖分开
└── docs/                     # 逐图索引、复现说明、版本边界、来源哈希
```

## 快速开始

在本目录执行；使用独立 Python 环境，不要覆盖已有实验环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[plots]'
python tools/verify_sources.py
python -m unittest discover -s tests -v
python assets/scripts/build_assets.py all
```

图中的 TeX 部分需要 TeX Live/MacTeX 的 `pdflatex`、`standalone`、TikZ、PGFPlots。
`all` 只执行已具备本地输入的图表生成，不重新跑大规模实验；具体覆盖见生成的 `generated/assets/all_build_report.json`。
所有输出默认放入被 Git 忽略的 `generated/`，不会覆盖保留的原始结果。

只核对 Figure 3 的数值，不装 LaTeX 也可以：

```bash
python assets/scripts/build_assets.py runtime --no-tex
```

## 核心算法

```bash
python -m pip install -e '.[native]'
# 先安装 Eigen3 开发头文件，再编译未经修改的官方 base.cpp：
python tools/build_hcp.py --eigen-include /path/to/eigen3
```

```python
from hmw import GaussianMixture, OfficialHCP, hmw

backend = OfficialHCP("build/hcp/base.so")
# means: (n,d), covariances: (n,d,d) SPD, weights: (n,) 严格正
source = GaussianMixture(source_means, source_covariances, source_weights)
target = GaussianMixture(target_means, target_covariances, target_weights)
result = hmw(source, target, backend)
print(result.distance, result.rows, result.cols, result.mass)
```

`src/hmw/` 是历史 Figure 3 路径的轻量分块封装，不包含 q=2、H6、球面后处理或 refinement。
新增的非法输入检查不改变有效 SPD 数据的运算，但**历史时间应由保留的原始脚本复测**，不能用新封装时间冒充。
当前论文的完整 canonical 算法与历史入口仍有差别，详见 [实现边界](docs/IMPLEMENTATION_STATUS.md)。

## 从哪里找

- [整篇论文逐图逐表索引与缺项](docs/ASSET_INDEX.md)
- [重绘、重跑实验、外部数据依赖](docs/REPRODUCTION.md)
- [Figure 3 数据、投影与计时说明](docs/FIGURE3_PROTOCOL_CN.md)
- [实现边界与测试记录](docs/IMPLEMENTATION_STATUS.md)
- [原始文件来源及 SHA-256](docs/SOURCE_MANIFEST.json)
- [第三方来源和许可](docs/THIRD_PARTY.md)

搜索历史被保留以交代 Figure 4 的挑选过程：这是探索后选择的展示案例，不是未经查看的独立优越性测试。

## 打包

```bash
python tools/package.py --output ../output/HMW_paper_code_20260926.zip
```

不打包 `.git`、缓存、编译产物、虚拟环境或临时重跑结果。保留的数据及参考图正常包含在内。
