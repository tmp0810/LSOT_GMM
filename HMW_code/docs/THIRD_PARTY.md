# 第三方来源与许可

## HCP

- 来源：<https://github.com/sherlockLitao/HCP>
- 提交：`930ecc98d2a6c6e884d96015919fed7de7c664f0`
- `src/base.cpp` SHA-256：`8f39fe76d99be69459b80612b5c6adb64176da6c9059d678a948b355cef0cddd`
- 原始源码/README/GPL-3.0 LICENSE 在 `third_party/hcp/`，未改动。
- 新编译入口 `tools/build_hcp.py` 仅替换原 setup 中机器专用 include 路径，不改排序/运输代码。

## DSMW / MSW

- 冻结提交：`5e6f32654cce84d064191ac0aaeb0d666c2b8a6f`
- 原实验文档给出的官方地址：<https://github.com/MoePien/sliced_OT_for_GMMs>
- 来源文件：原实验记录指定的官方 `sliced_mw.py`，快照位于 `third_party/dsmw/`。
- 预期 SHA-256：`35f73b312afc47e7347edc61c57f09f5df0fa3360707200e3d3d8cfb845e7069`
- 此快照不含完整 Git 历史或上游许可文件；原 benchmark 仍要求干净的官方 checkout。
- 原 MSW 函数引用未导入的 `sq_gaussian_wasserstein_distance_matrix`，历史 wrapper 通过复制函数 globals 补入对应一维平方代价，未更改算法和调用预算；见 `_original/benchmark_direct_svec_hcp_scaling_server.py`。
- 公共再分发前应补齐上游许可，不为第三方代码擅自声明新许可。

## CTDGMR

- 本地保留的官方源码归档 `third_party/ctdgmr/`。
- 原 README 指定 **CC BY-NC 4.0**，README 和 requirements 原样保留。
- 主文使用其 hard CTD-ISE 分配与 BFGS 密度重心；原 requirements 为历史版本，不应直接强制降级现有环境。

## GHAP / Gaussian Splatting

- 原实验固定 GHAP 提交：`27b3a581c861069cc97e5629e216dc3d1f42a98a`。
- 本包包含我们的适配/调用与原始 provenance 记录，不包含完整官方 checkout、pytorch3d、renderer 或训练数据。
- 上游环境、许可和数据授权需要分别保留；源码快照不改变它们的适用条件。

本仓库暂未给作者自有代码指定新的开源 LICENSE。用户确定公开发布策略前，请将它作为本地研究复现包使用。
