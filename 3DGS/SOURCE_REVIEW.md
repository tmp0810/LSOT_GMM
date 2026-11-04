# Rà soát source 3DGS trước khi triển khai

Ngày: 2026-11-04. Commit LSOT_GMM: c3a71f556bb51d23deed59cccef38a97296514e2.
Commit GHAP được HMW ghi nhận và đã dùng để đối chiếu: 27b3a581c861069cc97e5629e216dc3d1f42a98a.

Đây là source audit ở thời điểm trước triển khai. Khi đó chưa có PyTorch/POT và chưa chạy numerical tests. Các kiểm tra thực hiện sau đó được ghi riêng trong VALIDATION.md. Không có checkpoint scene và ảnh/camera trong upload đang đọc.

## Coverage

- HMW_code/README.md; mọi Markdown trong docs; SOURCE_MANIFEST.json.
- README của HCP và CTDGMR để hiểu phạm vi nguồn.
- Toàn bộ 25 file Python và 3 shell scripts trong experiments/scene_compression: 5.310 dòng Python.
- src/hmw, HCP base.cpp, build_hcp.py, verify_sources.py và test_hmw.py.
- lsot/plans.py, gaussians.py, projections.py, maps.py, optimized.py.
- param_proj/projections.py, distribution_proj/projections.py và gmmot.py; các test transport/projection liên quan.
- GHAP pinned source: README, arguments, root gmm_sampler, train_and_prune, Scene/GaussianModel/dataset/camera readers, renderer, render, metrics và utility liên quan.

Không coi các file được README nhắc tới nhưng không có trong cây Git là đã đọc. Không coi code CUDA renderer toàn bộ hoặc các thí nghiệm CTDGMR ngoài phạm vi 3DGS là đã review hết.

## Các nguồn quan trọng

| Source | Điều đã xác nhận |
|---|---|
| [IMPLEMENTATION_STATUS.md](https://github.com/tmp0810/LSOT_GMM/blob/c3a71f556bb51d23deed59cccef38a97296514e2/HMW_code/docs/IMPLEMENTATION_STATUS.md) | Nhánh 3DGS khác full Cholesky-HMW metric và khác các thí nghiệm reduction |
| [REPRODUCTION.md](https://github.com/tmp0810/LSOT_GMM/blob/c3a71f556bb51d23deed59cccef38a97296514e2/HMW_code/docs/REPRODUCTION.md) | Checkpoints/environment/data ngoài gói; Train 30k, Truck 15k, batch policy |
| [benchmark_pristine_ghap.py](https://github.com/tmp0810/LSOT_GMM/blob/c3a71f556bb51d23deed59cccef38a97296514e2/HMW_code/experiments/scene_compression/original/benchmark_pristine_ghap.py) | Pristine GHAP vs hilbert_optimized, seed42, warm-up, repeats, CUDA sync và writeout |
| [benchmark_pristine_lowratio.py](https://github.com/tmp0810/LSOT_GMM/blob/c3a71f556bb51d23deed59cccef38a97296514e2/HMW_code/experiments/scene_compression/original/benchmark_pristine_lowratio.py) | Retention từ config/environment, actual counts và algorithm record |
| [hcp_block_lloyd_batched.py](https://github.com/tmp0810/LSOT_GMM/blob/c3a71f556bb51d23deed59cccef38a97296514e2/HMW_code/experiments/scene_compression/original/hcp_block_lloyd_batched.py) | XYZ ordering, count-based groups, cap100 reps/block, incoming-cluster donor, eigen repair |
| [hcp_batched_solver.py](https://github.com/tmp0810/LSOT_GMM/blob/c3a71f556bb51d23deed59cccef38a97296514e2/HMW_code/experiments/scene_compression/original/hcp_batched_solver.py) | Frobenius geometry cost, arithmetic covariance update, empty repair và convergence |
| [prepare_pristine_finetune.py](https://github.com/tmp0810/LSOT_GMM/blob/c3a71f556bb51d23deed59cccef38a97296514e2/HMW_code/experiments/scene_compression/original/prepare_pristine_finetune.py) | Fresh Adam, common 30001..35000, geometry trainable, 4999 updates |
| [run_pristine_lowratio.sh](https://github.com/tmp0810/LSOT_GMM/blob/c3a71f556bb51d23deed59cccef38a97296514e2/HMW_code/experiments/scene_compression/original/run_pristine_lowratio.sh) | Hai scene, resolution khác nhau theo scene, pre/post render và metric pipeline |
| [blockwise_mw.py](https://github.com/tmp0810/LSOT_GMM/blob/c3a71f556bb51d23deed59cccef38a97296514e2/HMW_code/experiments/scene_compression/project_imports/application/gaussian_reduction/blockwise_mw.py) | Exploratory balanced fixed-capacity MW2, exact EMD, Bures update và guard |
| [gsplat_bridge.py](https://github.com/tmp0810/LSOT_GMM/blob/c3a71f556bb51d23deed59cccef38a97296514e2/HMW_code/experiments/scene_compression/project_imports/application/gaussian_reduction/gsplat_bridge.py) | Chunked Bures update, lookahead vs geometry-generating plan, opacity/SH/export |
| [hcp_partition_entry.py](https://github.com/tmp0810/LSOT_GMM/blob/c3a71f556bb51d23deed59cccef38a97296514e2/HMW_code/experiments/scene_compression/project_imports/application/gaussian_reduction/ghap_integration/hcp_partition_entry.py) | Float64 QR Cholesky từ raw scale/rotation; các partition-control branches |
| [main_coupling.py](https://github.com/tmp0810/LSOT_GMM/blob/c3a71f556bb51d23deed59cccef38a97296514e2/HMW_code/experiments/scene_compression/project_imports/application/gaussian_reduction/main_coupling.py) | q2/H6/T4 surrogate path, không phải entry của pristine HMW-local |
| [lsot/plans.py](https://github.com/tmp0810/LSOT_GMM/blob/c3a71f556bb51d23deed59cccef38a97296514e2/lsot/plans.py) | True lifted-cost selection, proportional ties, sparse plans, CPU MW2 wrapper |
| [lsot/gaussians.py](https://github.com/tmp0810/LSOT_GMM/blob/c3a71f556bb51d23deed59cccef38a97296514e2/lsot/gaussians.py) | Full Gaussian W2 edge prices, current batching và GMM validation |
| [GHAP gmm_sampler.py](https://github.com/taowang0105/GHAP/blob/27b3a581c861069cc97e5629e216dc3d1f42a98a/gmm_sampler.py) | Native KD, free capacities, empty-column deletion, donor toàn block, absolute floor |
| [GHAP train_and_prune.py](https://github.com/taowang0105/GHAP/blob/27b3a581c861069cc97e5629e216dc3d1f42a98a/train_and_prune.py) | Final iteration không optimizer step; densification và compaction gates |
| [GHAP dataset_readers.py](https://github.com/taowang0105/GHAP/blob/27b3a581c861069cc97e5629e216dc3d1f42a98a/scene/dataset_readers.py) | Sorted-name 1/8 holdout; exact camera counts phụ thuộc data |
| [GHAP metrics.py](https://github.com/taowang0105/GHAP/blob/27b3a581c861069cc97e5629e216dc3d1f42a98a/metrics.py) | PSNR/SSIM/LPIPS-VGG, per-view output, exception handling có thể che failure |

## Vấn đề cần xử lý trong triển khai mới

### 1. Source package chưa tự chứa đủ imports

Gói upload thiếu experiments/_original/official_hcp_backend.py và experiments/_original/nc_hot_bw. Eager imports trong application/common/__init__.py và gaussian_reduction/__init__.py khiến cả helper geometry đơn giản cũng có thể kéo theo dependency thiếu.

Không đưa toàn bộ legacy PYTHONPATH vào code mới. Port các primitive cần thiết với provenance và kiểm tra parity; hmw.hilbert.OfficialHCP là wrapper native có sẵn để dùng thay vì bịa một sorter HCP khác.

### 2. Kết quả historical không có trong upload

SOURCE_MANIFEST.json có 181 entries; 44 có mặt và khớp hash, 137 thiếu. README mô tả gói lịch sử rộng hơn upload hiện tại. Không dựng lại bảng chất lượng/timing từ các con số phỏng đoán.

### 3. Các thuật toán trong folder không đồng nhất

- Pristine report: XYZ-HCP + GHAP-cost Lloyd.
- Experimental blockwise MW: KD + frozen Voronoi capacities + Bures.
- Generic reducer: nhiều options, có moment matching và historical-plan guard.
- MainCouplingSolver: compact q2 surrogate, không exact Gaussian W2.
- FixedU2Mean: average NW plans theo một family khác LSOT.

Phải chọn rõ entry và không kết hợp các default từ các nhánh này mà không ghi trong config.

### 4. Nút thắt MW2 hiện tại

lsot.solve_mw2 gọi gmmot.GW2 với vòng lặp pairwise SciPy trên CPU. Nên giữ làm oracle nhỏ và tạo MW2 exact vectorized/chunked cho benchmark scene.

### 5. Nút thắt LSOT hiện tại

Current implementation đúng về full-cost selection và tránh dense K0*K1 cost. Nhưng mọi projection/entries được materialize cùng lúc và unique toàn bộ cặp; nguồn được tính roots lại theo lần gọi. Million-scale cần streaming, source caches, bounded cost chunks, và chunked source eigensolver.

### 6. Numerical geometry

Không dùng một floor tuyệt đối lớn cho mọi method rồi bỏ qua ảnh hưởng. Float32 covariance packing, floor cho covariance và floor cho sản phẩm S^(1/2)Sigma S^(1/2) có ý nghĩa khác nhau. Cần singleton/commuting/noncommuting regressions và reconstruction audit.

### 7. Donor có thể che lợi thế solver

Full-block donor trong blockwise_mw cấp phát cost all-pairs. Chunking giảm memory nhưng không giảm arithmetic. Shared incoming-plan donor là lựa chọn có thể mở rộng; phải ghi rõ khác native GHAP và ablate cùng rule cho các controlled methods.

### 8. Không nhầm heuristic update với variational LSOT optimizer

Bures update tối ưu fixed-plan Gaussian cost. Sau khi geometry thay đổi, finite-bank LSOT thay đổi order/plan nên descent không tự động đúng. Giữ plan cũ để ép monotonicity làm thay đổi algorithm semantics; cần đặt tên riêng hoặc kiểm tra objective LSOT thật.

### 9. Fine-tune phải match trạng thái

Cùng number of loops chưa đủ: checkpoint iteration, fresh Adam, parameter groups, spatial_lr_scale, camera order, exposure settings và densification gates cần match. HMW pristine cho geometry trainable; không copy freeze helper từ nhánh khác.

### 10. Reporting tính chất exact và count

Blockwise exact không phải global exact; native GHAP có thể trả ít hơn requested count. Mọi bảng ghi actual count, block policy và solver/backend. Không suy luận rendering speed hay PSNR chỉ từ transport cost.

## Kết quả kiểm tra đã thực hiện

- Local snapshot 87 file từ LSOT_GMM và 18 file GHAP khớp Git blob hash sau materialization.
- 25 file Python trong scene_compression parse AST thành công.
- 44 manifest files hiện có khớp SHA-256; không có mismatch trong những file đó.
- Không chạy HMW numerical suite đầy đủ vì dependency legacy được tests tham chiếu chưa có; không báo suite đó là passed.
- Không chạy compression/fine-tune/render: thiếu input scene, checkpoint và GPU environment cần thiết.
