# Kế hoạch thử nghiệm 3DGS: MW2 và LSOT

## 1. Mục tiêu và quyết định chính

Xây dựng một ứng dụng nén scene 3D Gaussian Splatting từ một checkpoint đã huấn luyện. So sánh MW2 chính xác trong từng block với min-LSOT dùng bốn họ projection Mix, SMix, B và B1D. Hỗ trợ thêm LSOT-avg cho cả bốn họ. Dùng GHAP nguyên bản và HMW-local làm baseline ứng dụng.

Giữ protocol dữ liệu, renderer, fine-tune và đánh giá theo nhánh HMW pristine. Trong nhánh MW2/LSOT, chỉ solver transport thay đổi; source, partition, số đại diện, khởi tạo, target masses, cập nhật geometry, appearance và lịch fine-tune dùng chung.

Hai phép đo riêng:

1. **Chất lượng nén scene:** cùng checkpoint, retention và đánh giá hình ảnh trước/sau fine-tune.
2. **Khả năng mở rộng solver:** tăng kích thước thực sự của cặp GMM trên Gaussian lấy từ scene; đo thời gian, bộ nhớ và lifted transport cost.

Không coi hai phép đo là một: tăng tổng số Gaussian nhưng giữ block nhỏ chỉ làm tăng số block, không bảo đảm speedup tăng theo N.

Tên ứng dụng nên là “3DGS compression with MW2/min-LSOT couplings”. Nếu chia block, ghi rõ **blockwise MW2/min-LSOT**. Exact EMD trong mỗi block không đồng nghĩa với nghiệm exact MW2 toàn scene.

## 2. Những điều source hiện tại xác nhận

### 2.1 Nhánh dùng cho so sánh HMW/GHAP

Entry points:
- HMW_code/experiments/scene_compression/original/benchmark_pristine_ghap.py
- HMW_code/experiments/scene_compression/original/benchmark_pristine_lowratio.py
- HMW_code/experiments/scene_compression/original/hcp_block_lloyd_batched.py
- HMW_code/experiments/scene_compression/original/hcp_batched_solver.py
- prepare_pristine*_finetune.py và run_pristine*.sh

HMW-local thực hiện:
- Tạo covariance từ scale/rotation gốc bằng float64, dùng QR để xây dựng Cholesky ổn định.
- Sắp Gaussian bằng HCP trên XYZ.
- Chia thứ tự thành M nhóm liên tiếp gần bằng nhau về **số atom**, không phải khối lượng.
- Gộp các nhóm liền kề thành block có tối đa 100 đại diện.
- Khởi tạo mỗi đại diện bằng trung bình mean và covariance có trọng số opacity trong nhóm.
- Chạy weighted Lloyd theo cost ||mu_i-mu_j||² + ||Sigma_i-Sigma_j||_F².
- Cập nhật covariance bằng trung bình số học, không cộng scatter giữa các mean và không phải barycenter Bures.
- Sửa cluster rỗng để giữ chính xác M.
- Chọn donor trong cluster cuối theo cùng cost GHAP; sao chép SH và raw opacity.
- Tối đa 100 cập nhật; dừng khi labels không đổi hoặc thay đổi objective đủ nhỏ.
- Batch 16 cho retention 0.1 và 0.01; batch 2 cho 0.001.

GHAP nguyên bản:
- KD partition mặc định depth=10.
- Chọn đại diện ngẫu nhiên trong mỗi block.
- Gán về các đại diện có cost nhỏ nhất; nếu cost bằng nhau thì chia đều mass giữa các minimizer.
- Target weights được cập nhật từ plan, không cố định.
- Xóa cột không nhận mass, nên output có thể ít hơn ngân sách yêu cầu.
- Donor là argmin cost trên toàn block theo từng cột, không nhất thiết thuộc cluster được gán.
- Tên hàm TensorKL không phản ánh công thức: đây là squared mean L2 + squared covariance Frobenius, không phải KL hay Gaussian W2.
- Export clamp eigenvalue ở mức tuyệt đối 1e-6.

Không gọi nhánh HMW-local trên là “full Cholesky-HMW distance”. Các module q=2/H6/T4 và fixed-U2 trong project_imports là những nhánh khác; chúng không được benchmark_pristine gọi để tạo nhánh hilbert_optimized.

### 2.2 Nhánh MW2 đã có trong gói upload

application/gaussian_reduction/blockwise_mw.py có:
- Chia KD block.
- Target capacities từ Voronoi của khởi tạo, rồi giữ cố định.
- Cost Gaussian W2 đầy đủ, tính trên GPU theo chunk.
- POT EMD trên CPU và kiểm tra primal/dual.
- Cập nhật bằng Gaussian Bures barycenter.
- Historical-plan guard: có thể giữ plan trước nếu plan mới đắt hơn.
- Donor toàn block theo cost GHAP.
- Export qua GHAP với absolute eigenvalue floor 1e-6.

Đây là reference tốt cho công thức và interfaces, nhưng không dùng nguyên xi làm thuật toán chính:
- Nó là nhánh exploratory, khác pipeline HMW-local đã báo cáo.
- Import kéo theo experiments.nc_hot_bw không có trong upload hiện tại.
- Donor và khởi tạo vẫn cấp phát pairwise arrays.
- Giữ plan từ vòng trước có thể khiến kết quả không còn là min-LSOT/avg-LSOT của geometry hiện tại.

### 2.3 Phạm vi dữ liệu thực có

Manifest của gói lịch sử liệt kê 181 file: 44 file có mặt và khớp SHA-256, 137 file không có trong cây Git hiện tại. Không có file hiện diện nào bị lệch hash trong kiểm tra này. Đây là so với manifest lịch sử, không phải 137 file đều cần cho thử nghiệm mới.

Thiếu các checkpoint Train/Truck, ảnh/camera scene, log kết quả scene lịch sử, manuscript snapshot và một số module legacy. Vì vậy chưa thể xác nhận lại số liệu bảng HMW hoặc tái chạy GPU chỉ từ upload này.

Source reader đã rà soát 25 file Python và 3 shell scripts của scene_compression (5.310 dòng Python), README, toàn bộ docs, HCP native core và các module LSOT liên quan. Đã đối chiếu thêm reader, model, sampler, renderer, train và metrics của GHAP.

## 3. Protocol dữ liệu và đánh giá

| Hạng mục | Thiết kế |
|---|---|
| Scene | Tanks & Temples: Train và Truck |
| Backbone | Train checkpoint 30.000; Truck checkpoint 15.000 theo script HMW |
| Retention rho | 0.1, 0.01, 0.001, tức 10%, 1%, 0.1% |
| Ngân sách | M=max(1,floor(rho*N)); ghi cả requested và actual |
| SH | Degree 3 |
| Resolution | Train: -r 1; Truck: -r -1 theo HMW |
| Ý nghĩa -r -1 | GHAP tự giảm chiều rộng về 1.600 nếu ảnh rộng hơn mức này |
| Split | --eval; COLMAP reader sort tên ảnh rồi lấy mỗi ảnh thứ 8 làm test |
| Đánh giá | Cùng camera IDs, kích thước ảnh, background, exposure và rasterizer cho mọi method |
| Trước fine-tune | Render và đo ngay checkpoint đã nén |
| Sau fine-tune | Cùng lịch 30.001 đến 35.000 |
| Optimizer fine-tune | Adam mới cho mọi method; geometry và appearance đều trainable |
| Số update | 5.000 vòng, 4.999 optimizer updates nếu giữ nguyên vòng lặp GHAP |
| Densification/compaction | Tắt trong fine-tune; assert số Gaussian không đổi |
| Metric | PSNR, SSIM, LPIPS-VGG bằng evaluator chung |
| Timing tham chiếu HMW | 1 warm-up + 3 lần đo, seed compression=42; median thời gian |
| Seed chất lượng mở rộng | Sau pilot, ít nhất 3 trial seeds chung nếu đủ tài nguyên; không nhầm với timing repeats |

Ảnh test không đi vào compression hay fine-tune. Hyperparameter cần chọn trước test cuối, hoặc chọn bằng validation tách từ tập train. Không chọn method/projection seed theo PSNR trên test.

Số view test phải được xác nhận trên bộ dữ liệu thực và lưu camera_manifest.json; source code một mình không xác nhận được số 38/32. Không tự gán những con số này vào output.

Nếu không lấy được backbone lịch sử, có thể huấn luyện backbone mới ở cùng ngân sách và dùng chung cho mọi method. Run đó phải ghi “retrained backbone”; không gọi là numerical reproduction của HMW.

Cần một run giữ nguyên checkpoint để kiểm tra renderer, camera và metrics trước khi chạy compression. Baseline không nén không được tính như một method ở cùng retention.

## 4. Formulation của nhánh MW2 và LSOT

### 4.1 Chuyển splat thành Gaussian atom

Với splat i:
- mean x_i là XYZ;
- scale s_i=exp(raw_scale_i);
- rotation R_i từ quaternion wxyz đã chuẩn hóa;
- Sigma_i=R_i diag(s_i²) R_i^T;
- opacity o_i=sigmoid(raw_opacity_i);
- source mass a_i=o_i/sum_k o_k.

SH và opacity dùng cho renderer được lưu riêng. a_i là mass của surrogate GMM, không phải renderer alpha; target capacity b_j cũng không được ghi trực tiếp vào raw opacity.

Tạo covariance từ raw scale/rotation bằng float64 để tránh lấy một covariance float32 đã mất tính SPD rồi thêm jitter lớn. Ghi thống kê eigenvalue/condition number và mọi sửa roundoff. Không thay bằng covariance diagonal hay Cholesky Euclidean cost.

Nếu có atom opacity đúng bằng 0, giữ mapping chỉ số gốc, ghi N_original và N_positive, xác định ngân sách từ checkpoint gốc và kiểm tra M<=N_positive. Không lặng lẽ sửa retention hoặc thêm fake mass.

### 4.2 Chia block và khởi tạo chung

Default cho bài nén scene: XYZ-HCP partition theo HMW, cap 100 đại diện/block.

- Sinh M nhóm liên tiếp theo số atom.
- Capacity b_j là tổng source mass trong nhóm khởi tạo.
- Mean/covariance khởi tạo dùng cùng quy tắc có trọng số cho mọi nhánh MW2/LSOT.
- Các capacity giữ nguyên trong toàn bộ outer iterations.
- Trong từng block, chuẩn hóa a và b theo mass block; khi cộng objective toàn scene phải nhân trả mass block.
- Lưu order, block IDs, initial group IDs, initial parameters và capacity hashes.

Tùy chọn KD partition là ablation chung cho MW2/LSOT, không đặt MW2 và LSOT vào các partition khác nhau trong phép so sánh solver.

Với block s có source mass A_s, objective bị ràng buộc block:
F_block=sum_s A_s F_s.
Đó là tối ưu trên tập plan không vận chuyển giữa các block. Nó không bằng objective không ràng buộc toàn scene.

### 4.3 MW2

Ở một block, source G_i=N(x_i,Sigma_i), target H_j=N(y_j,Lambda_j), capacities a,b cố định:

C_ij=||x_i-y_j||²+tr(Sigma_i)+tr(Lambda_j)
     -2 tr[(Sigma_i^(1/2) Lambda_j Sigma_i^(1/2))^(1/2)].

P_MW2=argmin_{P>=0, P1=a, P^T1=b} sum_ij P_ij C_ij.

Cần tính đúng cost này bằng vectorized/chunked float64 và giải EMD chính xác. Không benchmark MW2 bằng vòng lặp Python đôi và scipy.sqrtm hiện có trong gmmot.GW2 ở quy mô lớn: điều đó dễ phóng đại ưu thế runtime của LSOT.

POT EMD hiện dùng CPU. Geometry/cost có thể dùng GPU, nhưng tổng thời gian phải tính cả CPU solve và chuyển dữ liệu. Ghi rõ backend hybrid; không gọi EMD này là GPU-only.

Mọi nghiệm được kiểm tra marginals, finite/nonnegative masses, solver status, scaled primal-dual gap và dual feasibility. EMD có thể có nhiều plan tối ưu: test so objective và marginals thay vì đòi các phần tử plan luôn trùng.

### 4.4 LSOT-avg và min-LSOT

Giữ nguyên công thức và luật sampling của code hiện tại cho Mix, SMix, B, B1D. Một bank L=100 cố định trong mỗi trial và qua các outer iterations.

Với mỗi projection ell:
1. Chiếu từng Gaussian atom thành một số thực.
2. Giải 1D OT có trọng số.
3. Lift trở về các chỉ số Gaussian gốc, thu P^(ell).

Nếu có projected ties, dùng proportional/product lift theo fibers đúng như lsot.plans.lift_projection:
P_ij=Q_gh (a_i/A_g)(b_j/B_h) khi i thuộc fiber g, j thuộc fiber h.
Không thay bằng stable-sort arbitrary matching hoặc thêm jitter để né ties.

LSOT-avg: P_avg=(1/L) sum_ell P^(ell).
min-LSOT: ell*=argmin_ell <P^(ell),C>, P_min=P^(ell*).

Điểm bắt buộc: chọn min theo Gaussian W2 sau lift. Không chọn theo cost 1D projected và không thay C bằng cost GHAP.

Với cùng một cặp GMM và bank:
J_MW2 <= J_min <= J_avg
trong sai số số học. Quan hệ này không áp dụng để so output giữa những vòng tối ưu khác nhau hoặc kết luận trực tiếp về PSNR.

Chọn min riêng trong từng block là “blockwise min-LSOT”; các block có thể chọn projection khác nhau. Đây cũng không phải single-direction min-LSOT toàn scene.

### 4.5 Cập nhật đại diện chung

Giữ P cố định, đặt alpha_ij=P_ij/b_j:

y_j^+=sum_i alpha_ij x_i,
Lambda_j^+=argmin_{S SPD} sum_i alpha_ij d_BW²(Sigma_i,S).

Dùng Bures fixed-point iteration:
A_j(S)=sum_i alpha_ij (S^(1/2) Sigma_i S^(1/2))^(1/2),
S_next=S^(-1/2) A_j(S)^2 S^(-1/2).

Giữ arithmetic covariance update của GHAP chỉ ở baseline GHAP/HMW-local. Không trộn nó vào nhánh MW2 rồi gọi đó là tối ưu Gaussian W2.

Đây là Gaussian barycenter ở mỗi cột khi plan cố định. Nó không phải giải trực tiếp bài variational barycenter LSOT của nhiều GMM như thử nghiệm barycenter trước đây.

### 4.6 Outer iterations và tính đúng của tên method

Pilot có thể bắt đầu với tối đa 10 outer updates, 64 Bures inner iterations, inner tolerance 5e-4, theo ngân sách nhánh blockwise MW hiện có. Đây là cấu hình bắt đầu để kiểm tra, chưa phải setting tốt nhất.

Mỗi vòng cần:
- solve plan của geometry hiện tại;
- tính objective bằng ground cost thật;
- cập nhật geometry bằng plan đó;
- solve lại trên geometry mới để đánh giá đúng objective của chính method;
- lưu geometry, plan đã sinh geometry, plan mới được giải và objective tương ứng.

Với MW2, alternating minimization có cấu trúc descent nếu subproblems đủ chính xác. Với LSOT, sorting/lifting phụ thuộc geometry; không tự suy ra objective giảm sau khi solve lại.

Không mặc định kế thừa guard giữ plan lịch sử của blockwise_mw.py cho LSOT. Một plan cũ vẫn khả thi nhưng có thể không còn thuộc các lifted plans của bank trên geometry mới. Nếu thêm kiểu refinement này, đặt tên và log riêng.

Main pilot: ghi lại các iterate, báo objective thật và giữ iterate có objective thật thấp nhất đã đánh giá; không chọn checkpoint bằng test PSNR. Chạm iteration cap phải ghi capped, không ghi converged. Điều kiện converged cần đồng thời geometry change, refreshed-plan change, objective change và inner barycenter residual nhỏ; kiểm tra ổn định nhiều vòng để tránh kết luận từ một plateau.

Nếu oscillation xuất hiện: thử safeguarded geometry step/backtracking trên objective LSOT vừa solve lại, cùng quy tắc cho các nhánh controlled. Mọi lần solve phụ phải được tính thời gian. Không lặng lẽ đổi objective hay trả plan cũ dưới tên min-LSOT.

## 5. Appearance và export

Đây là một lựa chọn của ứng dụng, không do MW2/min-LSOT tự xác định.

Default đề xuất cho nhánh controlled:
- Với mỗi target, chọn donor có cost GHAP nhỏ nhất trong các source có P_ij>0 của plan dùng để tạo geometry đó.
- Tie-break theo original source index.
- Copy SH DC, SH rest và raw opacity từ donor.
- Cùng quy tắc cho MW2, min và avg; dùng sparse edges để tránh thêm all-pairs pass vào mọi method.
- Ghi tên rõ: incoming-plan donor, không gọi là upstream GHAP donor.

Hai ablation có ích:
1. Donor argmin trên toàn block theo đúng GHAP; stream cost theo chunk nhưng ghi rõ arithmetic vẫn O(nm).
2. Coupling mean của SH và activated opacity, rồi chuyển opacity về logit; không lấy trung bình logits.

HMW-local baseline giữ donor trong hard cluster cuối. GHAP baseline giữ donor toàn block nguyên bản. Đây là khác biệt của các baseline ứng dụng; phép so sánh nhân quả solver chỉ là nhóm MW2/LSOT dùng chung rule.

Export covariance thành log scales và quaternion wxyz, sửa orientation để det(R)=+1, kiểm tra reconstruction. Dùng cùng export policy cho toàn bộ nhánh controlled. Default là floor theo scale và độ chính xác float64 khi thật sự cần; thống kê số covariance bị chỉnh. Absolute floor 1e-6 của GHAP là ablation/export mode riêng vì có thể làm splat mỏng bị dày lên.

Trong đánh giá chính, mọi camera/image được render từ PLY đã round-trip, tránh một method được đo từ float64 tensors còn method khác qua PLY float32.

## 6. Tối ưu triển khai để mở rộng quy mô

### 6.1 Tái sử dụng đúng code LSOT

Có thể tái sử dụng:
- lsot.plans.SparsePlan, lift_projection và semantics của minimum_projected_plan.
- gaussian_pair_costs.
- Bốn projection samplers và công thức trong param_proj/distribution_proj.

Cần bổ sung:
- Stream theo block và projection batch thay vì giữ toàn bộ L(N+M) entries cùng lúc.
- Cache source roots, traces, covariance features và source orders vì source không đổi.
- Cache chỉ dùng trong một compression call; cold benchmark bao gồm phí xây cache.
- Tính cost chỉ cho các cạnh được yêu cầu; cache có giới hạn khi trùng cặp giữa nhiều projection.
- Với min: giữ best cost và projection index; tái dựng selected plan một lần nếu cần.
- Với avg: không dùng dense plan; xử lý coalesced sparse hoặc implicit sum đúng về số học.
- Batching eigensolver cho source validation, roots, Mix và B, không chỉ pairwise cost.
- Relative numerical safeguards cho splat rất anisotropic.
- Path xử lý ties phải giữ semantics product lift và giới hạn peak memory.

Ngoại lệ cần ghi trong độ phức tạp: khi không có projected ties, mỗi 1D plan có tối đa n+m-1 cạnh dương. Proportional lifting trên các tie fibers lớn có thể tạo nhiều cạnh hơn, thậm chí dense. Không tuyên bố bound sparse đó là vô điều kiện.

B có thêm các eigendecomposition theo Gaussian/projection; B1D chỉ cần projected variance/std trước bước cost selection. Vì vậy nên chạy B1D trước để kiểm tra pipeline, nhưng vẫn đo cả bốn họ.

### 6.2 Độ phức tạp cần trình bày đúng

MW2 có dense ground matrix n*m và chi phí EMD bổ sung; không gán một công thức O(n^3) chung cho mọi rectangular weighted problem.

LSOT có projection/sort và support pricing; ở d=3 và không ties, số edge visits khoảng L(n+m), nhưng khâu sort, coalesce, eigensolver và transfer vẫn có thể đáng kể.

Ở block có m=100 và L=100, số candidate edges LSOT có thể gần quy mô của toàn dense matrix. Do đó không hứa LSOT luôn nhanh hơn ở setting HMW mặc định.

Nếu giữ m<=100 khi tăng N toàn scene, local problem size gần cố định theo retention; speedup end-to-end không nhất thiết tăng. Đây là lý do cần solver-scaling experiment riêng.

## 7. Ma trận thí nghiệm đề xuất

### A. Kiểm tra environment và input

- Pin GHAP/HCP source commit và lưu file hashes.
- Kiểm tra PyTorch/CUDA/GPU architecture, renderer forward/backward, 3x3 eigh/QR và checkpoint restore.
- Có paths trong config, không hard-code /home/zpxiang hay /data/zpxiang.
- Chuẩn bị camera/split/checkpoint manifests và source counts.
- Render không nén, kiểm tra các view và metrics.
- Nếu checkpoint lịch sử không có, báo thiếu rõ và hỗ trợ backbone mới với provenance riêng.

### B. Pilot chất lượng

- Train trước, retention 10%, rồi 1%.
- Methods: MW2, min-LSOT-B1D, GHAP, HMW-local.
- L=100; shared XYZ-HCP blocks cap 100 cho nhóm controlled.
- Đánh giá pre; fine-tune cùng lịch rồi đánh giá post.
- Kiểm tra covariance/opacity, actual output counts, marginal residuals và objective histories.
- Sau khi pipeline đúng, mở Mix/SMix/B và avg.

Pilot chạy toàn scene cho hình ảnh. Tiny blocks dùng cho kiểm tra toán không được gọi là scene-quality results.

### C. Thử nghiệm chính

- 2 scenes x 3 retentions.
- Nhóm controlled: MW2 + 4 min-LSOT, thêm 4 avg khi cần ablation.
- Baselines: official GHAP và HMW-local.
- Baseline checkpoint không nén làm mốc chất lượng.
- L=100 mặc định; các thử nghiệm L dùng bank prefix chung.
- Timing: 1 warm-up và 3 measured repeats cho mỗi fixed configuration.
- Các trial seeds chất lượng tách khỏi các timing repeats.
- Pre/post dùng cùng camera list; ghi per-view và per-scene.

Nếu GHAP output thiếu ngân sách, giữ baseline native và ghi actual M; không thêm Gaussian trùng để đủ số lượng. Nếu sai khác lớn, thêm so sánh matched-actual-count chạy lại cả MW2/LSOT tại đúng M đó. Đó là một run riêng.

### D. Scaling trên Gaussian thật

Protocol D1 — tăng kích thước một OT problem:
- Nested subsets từ Gaussian của từng scene, cùng sampling seed và normalization cho mọi method.
- N đề xuất: 1k, 2k, 5k, 10k, 20k, 50k, 100k; chỉ chạy những mức data/resource thực có.
- M=floor(0.1*N) ở main scaling; 1% và 0.1% là các curve phụ nếu đủ tài nguyên.
- Cùng initial representatives và b tại từng N.
- Không chia thành ngày càng nhiều block nhỏ để giả là cùng “global scaling”.
- MW2: exact dense cost/EMD tới budget khả thi; vượt RAM/time thì ghi skipped_budget/timeout/OOM.
- LSOT chạy các mức lớn hơn chỉ chứng minh khả năng chạy; không tính speedup số tại N nơi MW2 không có timing.

Protocol D2 — block size trong application:
- Trên cùng full scene và retention, thử cap đại diện/block 100, 250, 500, 1000.
- Mọi nhánh controlled dùng cùng partition ở mỗi cap.
- Đo thời gian compression, bộ nhớ và chất lượng để thấy tradeoff locality/scalability.
- Chỉ chạy sau pilot, không tự mặc định tăng cap sẽ tốt hơn.

L ablation: 25, 50, 100, 200 với nested projection bank, dùng dữ liệu validation hoặc một grid cố định đã công bố.

## 8. Timing và metric output

Tách tối thiểu:
- source extraction / covariance construction;
- partition / common initialization;
- source preparation / source projection sorting;
- target projection sorting;
- dense ground cost + CPU EMD cho MW2;
- lifted-plan construction + full-cost selection cho min;
- Bures geometry update;
- appearance transfer và export conversion;
- total compression wall time;
- fine-tune process wall time;
- render-only timing nếu đo FPS.

GPU phải synchronize tại ranh giới thời gian. Có cả CUDA allocated/reserved peak và CPU peak RSS của worker riêng; không gọi một ngưỡng dense matrix là tổng peak memory.

Cold-call result và amortized/repeated-target result ghi riêng. Source preprocessing được tính một lần trong total compression, không loại khỏi LSOT trong khi MW2 phải tính lại.

FPS, nếu có, dùng warm-up và GPU timing trên đúng views/resolution, loại đọc/ghi PNG. Không suy FPS chỉ từ số Gaussian.

Bảng chính:
Scene | Retention | Method | N_out | Compression(s) | PSNR↑ | SSIM↑ | LPIPS↓.
Post-finetune là một bảng/khối rõ ràng; pre-finetune phải được lưu và trình bày vì fine-tune có thể che khác biệt compression. Fine-tune time nằm cột phụ hoặc appendix.

Plot:
- total solver time vs N (log-log);
- speedup MW2/min-LSOT tại những N cả hai đều hoàn thành;
- CPU/GPU peak memory vs N;
- lifted cost gap với MW2 trên cùng input;
- PSNR/LPIPS vs retention;
- timing breakdown;
- convergence/objective diagnostics.

Ảnh:
- Cùng view, resolution và crop cho các method.
- GT, checkpoint không nén, GHAP, HMW-local, MW2 và các min-LSOT.
- Ảnh pre và post tách rõ; không dùng khác view để làm một method đẹp hơn.
- B1D có thể là panel chính đã chọn trước; toàn bộ projection còn lại vẫn có output/appendix.

## 9. Cấu trúc folder sẽ triển khai

Root folder là 3DGS. Package Python nội bộ dùng tên hợp lệ lsot3dgs; không dùng cú pháp import 3DGS.

| File / nhóm file | Trách nhiệm |
|---|---|
| 3DGS/README.md | Hướng dẫn local/server/Colab, dependencies, output và protocol |
| 3DGS/IMPLEMENTATION_PLAN.md | Kế hoạch này |
| 3DGS/SOURCE_REVIEW.md | Code đã đọc, mapping, giới hạn và nguồn |
| 3DGS/configs/smoke.json | Tiny numerical/GPU smoke |
| 3DGS/configs/hmw_protocol.json | Scene, retention, paths, split và fine-tune |
| 3DGS/configs/scaling.json | N,M,L, time/RAM budgets, repeat policy |
| 3DGS/setup_env.sh | Cài dependency và build extension có kiểm tra commit |
| 3DGS/prepare_data.py | Kiểm tra/chuẩn bị dataset và checkpoint, viết manifest |
| 3DGS/preflight.py | GPU/renderer/checkpoint/camera sanity checks |
| 3DGS/run.py | Orchestrator có resume dựa trên config/input/code hashes |
| 3DGS/compress.py | Chạy một scene/method/retention và lưu PLY + diagnostics |
| 3DGS/finetune.py | Adapter gọi chung official loop; reset optimizer |
| 3DGS/evaluate.py | Pre/post render và metrics kiểm tra đầy đủ view |
| 3DGS/benchmark.py | Controlled solver scaling, timing và resource budgets |
| 3DGS/summarize.py | CSV/JSON, không bỏ qua run lỗi một cách im lặng |
| 3DGS/plot.py | Quality, speed, memory, convergence, image panels |
| 3DGS/lsot3dgs/model_io.py | Checkpoint/PLY/scale/quaternion/opacity conversion |
| 3DGS/lsot3dgs/partition.py | Shared HCP/KD partition và exact allocation |
| 3DGS/lsot3dgs/initialization.py | Initial groups, frozen capacities và hashes |
| 3DGS/lsot3dgs/geometry.py | Chunked SPD/root/Gaussian W2/Bures primitives |
| 3DGS/lsot3dgs/plans.py | Common sparse/implicit plan contract |
| 3DGS/lsot3dgs/solvers/mw2.py | Vectorized exact cost + audited EMD |
| 3DGS/lsot3dgs/solvers/lsot.py | 4 projections x min/avg; cache và streaming |
| 3DGS/lsot3dgs/reduction.py | Shared outer loop và genuine objective diagnostics |
| 3DGS/lsot3dgs/appearance.py | Incoming donor, full-block donor, coupling mean |
| 3DGS/lsot3dgs/baselines.py | Official GHAP và port HMW-local đã kiểm tra parity |
| 3DGS/lsot3dgs/timing.py | CUDA sync, wall timing, memory, repeats |
| 3DGS/tests/ | Numerical, parity, boundary và integration tests |
| 3DGS/colab.ipynb | Launcher sau khi CUDA environment được kiểm tra thực tế |

Các entry point dự kiến:
- python 3DGS/preflight.py --config 3DGS/configs/hmw_protocol.json
- python 3DGS/run.py --config 3DGS/configs/hmw_protocol.json --methods mw2 min-B1D ghap hmw-local --scenes train --retentions 0.1
- python 3DGS/benchmark.py --config 3DGS/configs/scaling.json
- python 3DGS/plot.py --results-dir results/3dgs

Các lệnh trên mô tả CLI dự kiến, chưa phải script đã có thể chạy.

Tái sử dụng utilities LSOT theo API có sẵn. Nếu cần sửa chunking/cache trong core, giữ mặc định/backward compatibility và chạy regression cho các phép chiếu/transport cũ. Không phụ thuộc vào chuỗi import application/experiments legacy bị thiếu.

## 10. Output và khả năng resume

Một run được nhận diện bởi scene, checkpoint hash, input/split hashes, retention, actual M, method/projection, bank hash, seed, partition/init/capacity hashes, covariance/export/appearance modes, solver budgets và code version.

Output tối thiểu:
- resolved_config.json, provenance.json, environment.json, camera_manifest.json;
- source_stats.json, partition/init/capacity metadata;
- compression.json, history.jsonl, timings.jsonl;
- pre/point_cloud/iteration_1/point_cloud.ply và cfg_args;
- pre/test/<method>/renders, gt, results.json, per_view.json;
- post/start.pth, point_cloud/iteration_35000/point_cloud.ply;
- post/results.json, per_view.json, train_wall.txt;
- quality.csv, runtime.csv, scaling.csv và figures/.

Đọc results.json không đủ để đánh dấu thành công: cần đủ views, metric hữu hạn, PLY đúng count và matching config/hash. metrics.py upstream bắt exception rộng rồi in “Unable to compute metrics”; wrapper phải phát hiện tình huống này và đánh dấu failed, không coi exit code 0 là thành công.

Resume không tái dùng kết quả cũ khi config, source, projection bank, code hoặc appearance rule thay đổi. File completion marker chỉ viết sau khi kiểm tra xong.

## 11. Tiêu chí nghiệm thu trước khi chạy lớn

1. PLY/checkpoint round-trip: covariance, quaternion, SH và opacity giữ đúng ý nghĩa; r=1 identity baseline render không đổi trong tolerance.
2. Các covariance hợp lệ/finite; numerical repairs được định lượng.
3. Sparse plan đúng cả row/column masses, không có mass âm.
4. Streaming LSOT trùng reference nhỏ ở cost, chọn direction và plan; có cases projected ties và unequal weights.
5. J_MW2<=J_min<=J_avg trên cùng GMM, bank, partition và capacities.
6. Exact EMD có certificate; cost vectorized khớp GaussianW2 reference nhỏ.
7. Bures update: singleton giữ nguyên; diagonal/commuting covariances khớp nghiệm ((sum_i alpha_i sqrt(diag Sigma_i))²); noncommuting case có residual và objective đúng.
8. Geometry, appearance và plan IDs không bị lệch vòng; giá trị objective không dùng cost cũ.
9. Fixed capacities không đổi qua vòng; không lẫn free-weight CTD.
10. Không cấp phát dense N*M trong min path; tie/avg path ghi rõ chi phí thật.
11. GHAP native và HMW-local không bị giả là cùng budget nếu actual counts khác.
12. Fine-tune dùng cùng fresh Adam, trainable parameter groups, camera schedule và số updates; không densify/prune.
13. GPU/CPU parity ở bài nhỏ và smoke CUDA renderer/backward trên environment thật.
14. Whole-scene results chỉ báo khi đầy đủ pre/post metrics; capping/OOM/timeout được giữ trong bảng log.
15. Các thay đổi core không làm regression các thử nghiệm trước.

## 12. Thứ tự triển khai

- Giai đoạn 1: environment/input manifests, model IO, renderer identity check; port baseline native GHAP/HMW-local đủ độc lập để chạy.
- Giai đoạn 2: common partition/init/plan contracts, vectorized MW2, streaming min-B1D; numerical acceptance tests.
- Giai đoạn 3: shared Bures update/appearance/export, Train 10% pre/post pilot và convergence audit.
- Giai đoạn 4: bốn projection và avg, hai scene x ba retention, quality tables.
- Giai đoạn 5: global solver scaling, block-cap ablation, timing/memory/quality plots và Colab launcher.

Chỉ sau phép đo mới kết luận speedup và chất lượng. Mục tiêu khoa học là chỉ ra vùng kích thước mà min-LSOT đạt tradeoff tốt; không mặc định mọi retention, projection hay block size đều thắng MW2.
