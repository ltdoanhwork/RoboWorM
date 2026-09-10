# Review story, contribution và novelty — 2026-09-09

**Đánh giá:** hướng bài có giá trị, nhưng draft đang claim rộng hơn chứng cứ trong repo. Trục mạnh nhất: **kiểm tra cấu trúc robot trong từng view + tính nhất quán giữa các view, định vị lỗi theo thời gian, rồi dùng để đo đánh đổi tốc độ–fidelity**. Điểm yếu lớn nhất hiện tại: validation của chính evaluator và tính nhất quán của experimental protocol. Thêm accelerator hoặc thêm số video chưa giải quyết hai điểm này.

Review dựa trên source, metadata, tài liệu, output hiện có và đối chiếu primary papers. Không chạy generation, training hay sửa pipeline. Số đếm dưới đây là snapshot file-system; không thay thế kiểm tra decode, dedup và provenance từng rollout. Những đề xuất thí nghiệm là việc cần làm, không phải kết quả đã có.

## 1. Những claim cần sửa trước

| Claim trong draft | Chứng cứ repo / vấn đề | Sửa đề xuất |
|---|---|---|
| Tất cả là “action-conditioned robotic world models” | `metrics/sim2real/interaction_metrics.py:623` tách rõ `action` và `text`; DreamGen video2world_lvg nhận first frame + text. Các script DreamGen multiview cũng dùng grid + prompt. | Gọi chung “robotic video world models”; phân track action-conditioned và image/text-conditioned. Chỉ dùng action-following claim cho nhánh nhận action thật. |
| “independently generated views” | Ctrl-World sinh latent strip rồi split view; DreamGen sinh grid. `scripts/run_humanoid_multiview_dreamgen_dense.sh:1` mô tả 2×2 grid. | “different generated views of the same episode”, hoặc “jointly generated views”. Independence chỉ đúng nếu thực sự generate từng view riêng. |
| “Markovian and non-Markovian rollout regimes” | `scripts/classify_markovian_qwen.py:43` phân loại mô tả task: atomic vs sequential/requires history. Không phải can thiệp history/reset của generator. | Dùng “atomic and sequential task subsets”. Multi-step không tự động là non-Markovian; Markov property phụ thuộc state representation. |
| “full factorial” và adaptation held constant | [Canonical matrix](gpu_agent_fill_missing.md:36): DreamDojo chỉ SV; Ctrl-World chỉ MV; DreamGen cả hai. Source, robot và conditioning cũng khác. | “12 evaluation strata with model-specific coverage”. 12 strata dữ liệu có thể đủ; không suy ra factorial model × view hay causal effect của view. |
| “three embodiments … one known robot body” | [Execution notes](../metrics/embodied_exec/README.md:46): humanoid SV là GR1; MV là Airbot MMK2, mobile dual-arm robot. | “three embodiment categories”; liệt kê robot cụ thể. Không coi SV/MV humanoid là cùng robot thay camera. |
| “fixed set of calibrated viewpoints” | Repo có wrist camera, ego camera; execution notes ghi thiếu intrinsics cho nhánh reconstruction. Không có bằng chứng calibration đồng nhất toàn bộ benchmark. | “known camera layouts”; chỉ ghi calibrated/static cho subset thực sự có calibration/static cameras. |
| “Every metric … without a vision-language model … segmentation, tracking, geometry alone” | [Metric definition](phyworld_metrics.md:10) có PhyQual dùng CLIP/GroundingDINO; dynamics dùng SEA-RAFT/VFIMamba. Interaction validity còn có CLIP identity gate. | Định nghĩa phạm vi “no generative VLM judge in the core scores”. Công khai pretrained perception, text-based localization, gates và data curation. Không gọi toàn pipeline model-free. |
| “motion follows the depicted interaction” | ActorDynamics đo foreground–background flow; MotionContinuity đo interpolation. Cả hai chưa đo contact hay force transmission. | Gọi “motion diagnostics”; muốn claim interaction phải validate interaction tier riêng. |
| “identical post-training protocol” | Script/config có checkpoint, số bước, source, model family và conditioning khác nhau; chưa có frozen comparison manifest chứng minh identical. | “documented model-specific adaptation protocols”; nêu rõ phần nào matched, phần nào khác. |

**Hệ quả:** không nên chữa bằng wording đơn lẻ. Bài cần định nghĩa đúng đơn vị so sánh: trong cùng model/checkpoint/source/view/conditioning/horizon, accelerator thay đổi gì? So sánh across models là descriptive, trừ khi các yếu tố còn lại đã được kiểm soát.

## 2. Repo hiện có gì, chưa có gì?

### Corpus lớn; số draft chưa phải số hoàn thành

Đếm lại theo quy tắc `units()` và canonical cells trong [audit repo](gpu_agent_fill_missing.md:194), ngày review:

| Method | Output units được tính, cap theo target từng cell | Target |
|---|---:|---:|
| Dense | 3,131 | 3,166 |
| DiCache | 3,166 | 3,166 |
| WorldCache | 3,166 | 3,166 |
| FasterCache | 2,233 | 2,361 |
| SiTo | 2,290 | 2,410 |
| iTM | 2,290 | 2,410 |
| PISA | 3,046 | 3,166 |
| Radial | 3,046 | 3,166 |
| SVG1 | 3,046 | 3,166 |
| SVG2 | 3,046 | 3,166 |
| **Tổng** | **28,460** | **29,343** |

883 units chưa đủ target theo phép đếm này. Một số cell có nhiều output hơn target, nên script cap lại; **đây chưa phải 28,460 unique valid rollouts**. Cần join theo episode, checkpoint, seed và config để loại rerun/variant. File tồn tại không chứng minh decode được, đúng thời lượng hay metric đã tính.

`26,177 + 3,166 = 29,343` đúng về kế hoạch. Chưa nên viết như completed experimental result. `1,628` ở P4 và `1,608` ở contribution cũng phải thống nhất; execution notes ghi index 1,608, canonical DreamGen target là 1,605 sau missing-camera exclusions. Chưa kiểm chứng độc lập toàn bộ 60 tasks và 3,811 source videos.

SiTo/iTM không có DreamDojo integration trong canonical sweep; FasterCache bị loại khỏi Ctrl-World vì CFG-off no-op theo audit. Vì vậy “nine methods across three models” cần đi cùng applicability matrix, không ám chỉ đủ 27 method–model pairs.

### Metric artifacts chưa tương ứng quy mô generation

- [phyworld_tables.json](../sampling_dataset/phyworld_tables.json) chỉ chứa **một cell, bốn episodes, dense và WorldCache**. Đây là pilot đã lưu, chưa phải bằng chứng full-sweep diagnosis. Có thể còn kết quả ở nơi khác; review chưa xác minh được.
- [metrics.multiview.json](../webpage/labeling/metrics.multiview.json) có **300 machine-scored clips**.
- [metrics.segment.json](../webpage/labeling/metrics.segment.json) có **291 videos, 8,093 segments**, không phải 300 videos. Jerk chỉ tham gia detector của single_arm; bimanual/humanoid dùng rigidity. Contribution không nên ám chỉ hai body detector đều được dùng ở mọi embodiment.
- [docs/phyworld_metrics.md](phyworld_metrics.md:83) ghi AUC 0.987. Review chưa tái lập được từ human votes. `sampling_dataset/data_for_web/labels.json` hiện rỗng; labels có thể nằm ở backend ngoài repo. `machine_score`, `band`, `violated` trong catalog **không phải human labels**.
- Source release chưa khép kín: `scripts/run_phyworld.sh` được tài liệu dẫn nhưng không có ở checkout; `metrics/phyworld/` hiện chỉ thấy `__pycache__`. Tìm kiếm chưa định vị được source tính BodyRigidity/ViewGeometry, dù có JSON kết quả và builder catalog. Cần đưa evaluator, calibration và script tái lập về repo hoặc pin external dependency rõ ràng.
- [embodied_exec](../metrics/embodied_exec/README.md:12) ghi chưa chạy trên generated video, còn thiếu pose decoder. [real2sim](../metrics/real2sim/README.md:3) ghi segmentation chưa đủ tin cậy để report. Không đưa hai nhánh này thành completed contribution.

## 3. Novelty nên đặt ở đâu?

**Không đủ mới nếu đứng riêng:** robot setting cố định; algorithmic scoring; nhiều metric; per-dimension reporting; post-training; no VLM judge. EWMBench đã lập luận embodied scenes có cấu trúc và dùng trajectory metrics cùng semantic evaluation. [EWMBench, Introduction và §3](https://arxiv.org/html/2505.09694v1).

RoboPhys-3D tái dựng main-perspective video; điểm này hỗ trợ phân biệt với simultaneous cross-camera checking. Nhưng họ có 50 metrics, per-dimension analysis, GT reconstruction calibration và execution evaluation. Câu “only whether that view is internally coherent” mô tả sai phạm vi; câu “none … per-property diagnosis” cũng không giữ được. [RoboPhys-3D, §3.4–4.1](https://arxiv.org/html/2608.28718v1).

MiraBench đã tách physics adherence, action following và optimism bias; kinematic pipeline vẫn có VLM quality gate. Đây là đối thủ gần cần đối chiếu. [MiraBench, §D–E](https://arxiv.org/html/2605.29360v1). RoboWM-Bench dùng embodied execution, nên downstream executability cũng không phải khoảng trống chưa ai chạm tới. [Project chính thức](https://robowm-bench.github.io/RoboWM-Bench/).

Physics-IQ dùng controlled physical continuations và reference-based scoring. Vì vậy P3 không thể kết luận ngoài robot “only remaining signals” là appearance và model judgment. [Physics-IQ, repo chính thức](https://github.com/google-deepmind/physics-iq-benchmark).

**Novelty khả thi nhất là tổ hợp có chứng minh:**

1. **Cross-view evidence bổ sung thông tin mà từng view riêng không có.** Ví dụ hai view riêng đều hợp lý, nhưng khác contact state hoặc thời điểm chuyển động. Cần baseline single-view đạt tốt mà cross-view vẫn phát hiện lỗi.
2. **Body consistency được calibration và định vị lỗi theo segment.** Phải chỉ ra metric đo lỗi robot, không chủ yếu đo tracker/depth estimator thất bại.
3. **Acceleration diagnosis theo từng thuộc tính tại mức speedup tương đương.** Kết quả cần là trade-off hoặc ranking reversal cụ thể, không chỉ thêm cột metric vào bảng latency.

Không cần cố gọi từng công thức là metric hoàn toàn mới. Rigidity, jerk, flow EPE và epipolar geometry đều có lịch sử. Contribution mạnh hơn nếu là **measurement protocol được validate, có uncertainty/coverage và tạo ra kết luận mới**. Review này chưa đủ để chứng nhận claim “first”; nên bỏ claim tuyệt đối cho đến khi có related-work matrix đầy đủ.

## 4. Review story P1–P6

**P1 — giữ động cơ, thu hẹp cách nói.** “Stand-ins for the robot itself” chưa chính xác: world model thay interaction dynamics/environment rollouts. Dùng “surrogates for robot–environment interaction”. Policy evaluation, data generation, planning là motivation; benchmark offline hiện tại chưa chứng minh đáp ứng cả ba ứng dụng. “No longer built mainly” là claim xu hướng không cần thiết. “Increasingly used…” đủ.

**P2 — rút còn khoảng nửa độ dài.** Nhánh VLM đang lấn át contribution. Position bias, self-preference, weak-judge, ensemble failure cần citation riêng và không tự động chuyển từ text/image judge sang physics video judge. Không có nguồn đúng setting thì bỏ. Thay bằng hạn chế đo lường cụ thể: khó xác định lỗi nằm ở đâu, cần validation và không trực tiếp kiểm tra agreement giữa cameras.

PSNR/SSIM/LPIPS cũng không phải aesthetic metrics; chúng đo similarity với reference. Frozen rollout có thể được lợi khi background áp đảo hoặc ở temporal smoothness metric, nhưng không mặc nhiên được PSNR ưu ái trên mọi dynamic GT. Dùng wording có điều kiện và minh họa bằng frozen baseline.

**P3 — giữ insight, bỏ false dichotomy.** Known embodiment và camera layout cho phép đặt kiểm tra cụ thể; không bảo đảm scene đầy đủ quan sát được. Occlusion, metric scale, moving camera và tracker identity vẫn là uncertainty. Đổi “checked directly rather than interpreted” thành “tested with explicit, auditable consistency checks”.

**P4 — nêu hai trục chính trước.** Body integrity và simultaneous cross-view coherence là tâm bài. Dynamics là supporting diagnostics; dense-reference là acceleration analysis. Không liệt kê bốn nhóm như bốn physical laws ngang nhau.

**P5 — nối bằng câu hỏi stress test.** “These gains in fidelity…” không có antecedent: benchmark chưa cải thiện generator. Đổi thành “Beyond evaluating dense generation, the same diagnostics expose the fidelity cost of inference acceleration.” Caching smears motion/token reduction thins arm hiện là hypotheses; chưa được viết như finding trước khi có matched sweep.

**P6 — ba contributions, mỗi cái có một bằng chứng.** Benchmark/protocol → manifest và coverage; metrics → human/controlled validation; diagnosis → Pareto/ranking result. Số video là quy mô, không thay bằng chứng novelty.

## 5. Bổ sung gì trước? Thứ tự theo giá trị cho bài

### Ưu tiên 0 — protocol và khả năng tái lập

Tạo frozen manifest: `source_episode_id`, source dataset, robot ID, task, task subset, model/checkpoint hash, conditioning type, seed, action source, prompt source, history/reset policy, view layout, FPS, duration, resolution, accelerator config, output validity, metric version và coverage.

Phân rõ GT episodes, camera videos, model rollouts, tiled output videos, segments và acceleration variants. Không cộng cùng episode nhiều lần rồi gọi independent episodes. Audit train/calibration/test overlap theo source episode; mọi view và mọi variant của cùng episode phải cùng split.

Kiểm tra prompt có được tạo từ future GT không, action có dùng latent encoder nhìn future frames không. Nếu có, ghi conditioning là privileged/offline reconstruction; không đánh đồng với online planning.

### Ưu tiên 1 — validate evaluator; đây là add-on quan trọng nhất

**A. Human validation độc lập.** Tận dụng 300 geometry clips và 291 body clips. Tách calibration/test theo episode hoặc task, không random segments. Ít nhất 3 raters/clip hoặc segment subset đã định trước; giấu accelerator và machine band. Xuất human votes, adjudication, agreement và script join IDs.

Body rubric hiện gộp blur, color, camera shake và geometric deformation; geometry rubric cũng gộp màu và camera stability. Nếu metric đo rigidity/pose, nên có nhãn property-specific bên cạnh overall label. AUC trên “good vs bad” bỏ medium dễ hơn bài toán thực; report thêm ordinal correlation hoặc detection ở cả medium và bad. Bootstrap theo episode, không coi 8,093 segments độc lập.

**B. Controlled counterexamples + negative controls.** Mỗi perturbation có severity sweep và benign control:

| Kiểm tra | Vi phạm chủ đích | Control không nên bị phạt |
|---|---|---|
| Body rigidity | Stretch một link, thêm/mất bộ phận | Articulation hợp lệ, đổi góc nhìn |
| Body temporal continuity | Teleport hoặc spike | Chuyển động nhanh nhưng trơn, camera motion đã bù |
| Cross-view timing | Delay chỉ một view; ghép sai episode cùng task | Hai camera đồng bộ dù khác appearance |
| Cross-view geometry | Dịch/đổi một actor trong một view | Occlusion hợp lệ, ít overlap → abstain |
| Motion coverage | Freeze trong đoạn GT có chuyển động | Pause hợp lệ trong task |
| Robustness | — | JPEG, blur nhẹ, illumination và resize giữ physics |

Report detection-vs-severity, false positives trên real GT và metric coverage. Một unknown track không được tính thành physics pass. Cần phân biệt input không quan sát đủ với robot biến mất do generator; nếu bỏ hết clip khó, accelerator xấu có thể được điểm cao hơn vì evaluator abstain nhiều hơn.

**C. Ablation của perception.** Chấm một subset với keypoints/masks sửa thủ công; so ranking trước/sau. Thử backend hoặc tracking seed khác trên subset. Đây là cách đo phần lỗi do evaluator. Real GT phải đi qua cùng pipeline và cùng resize/FPS; cùng pipeline giúp ước lượng bias, không đảm bảo bias triệt tiêu trên generated distribution.

### Ưu tiên 2 — chứng minh giá trị riêng của cross-view

Thiết kế tập “individually plausible, jointly inconsistent”. So best single-view metric, pooled per-view metrics, rồi thêm cross-view metric trên cùng held-out clips. Report phần cải thiện detection và localization, không chỉ AUC tổng.

Tách correspondences trên robot/object và background. Background chiếm ưu thế có thể làm geometry tốt dù robot mỗi view hành động khác. Report visible overlap/inlier coverage và camera type. Essential/fundamental matrix ổn định mới là compatibility evidence, chưa chứng minh chung một dynamic 3D world. Views có thể đồng ý với nhau nhưng cùng sai so với action hoặc GT.

Đây là figure/microbenchmark có khả năng nâng novelty hơn thêm hàng nghìn clips ngẫu nhiên.

### Ưu tiên 3 — acceleration tạo kết luận có ích

Chọn matched subset đủ phủ model families; thử 3–4 mức aggression/method và nhiều seeds trên subset. Giữ checkpoint, conditioning, output grid, physical duration, denoising schedule cố định. Đo latency end-to-end, denoising latency, hardware/batch size và actual accelerator activation. Dùng dense-repeat để ước lượng numerical/stochastic variability.

Report ba thứ riêng: absolute violation rate, paired change so với dense, và dense-reference fidelity. Dense không phải physical upper bound: accelerator có thể sửa một lỗi tình cờ. Không ép score accelerated ≤ dense. Câu “dense is NR ceiling” trong tài liệu hiện tại cần sửa trước khi thành evaluation rule.

Plot Pareto speedup–violation theo property. So method tại matched speedup hoặc matched error budget, không chỉ default hyperparameters. “Family X degrades first” chỉ đứng được khi thứ tự này bền qua configs, seeds và model cells; nếu không, report conditional finding.

Định nghĩa thực dụng: **maximum tested speedup under a predeclared fidelity budget**, kèm confidence interval và coverage floor. Tránh gọi “safe speedup”: metric pass không đảm bảo robot safety. Nếu fixed sampled config, chỉ gọi “best tested feasible configuration”, không suy ra optimum liên tục.

### Ưu tiên 4 — bổ sung interaction nhỏ, có điều kiện

Nếu muốn giữ “coherent physical world”, Body + View chưa đủ: robot còn nguyên và views đồng ý nhưng object vẫn trượt khỏi gripper hoặc chuyển động không có support. Repo đã có contact/grasp-follow/object-integrity và negative controls ở [sim2real](../metrics/sim2real/README.md).

Chọn một subset static-camera, object track rõ để validate contact/grasp-follow; không cần full simulator. Scope “observable interaction consistency”, tránh gọi full Newtonian validity. Object chuyển động khi không tiếp xúc có thể là free fall/inertia hợp lệ; 2D mask overlap cũng không chứng minh 3D penetration. Gate theo phase/task và report unknowns.

Nếu thời gian hạn chế: giữ interaction như focused case study, thu hẹp claim bài về structural/spatiotemporal fidelity. Không kéo real2sim/force feasibility thành headline khi upstream tracking/scale còn chưa ổn.

### Ưu tiên 5 — một liên hệ downstream, nếu đủ nguồn lực

Để nối P1 với experiments: kiểm tra metric chọn giữa hai rollouts/accelerators có phù hợp với independently annotated grasp/placement outcome trên held-out subset không. So với LPIPS/SSIM và một VLM baseline cùng samples. Đây là decision utility cho chọn rollout, chưa phải chứng minh policy transfer hoặc execution success.

Thực thi robot/simulator và học pose decoder có giá trị nhưng tốn nhiều hơn; repo ghi rõ decoder chưa có. Không cần mở nhánh lớn này để cứu paper nếu Body/View validation đã mạnh.

## 6. Những giới hạn metric cần nói thẳng

1. **Jerk cao là kinematic anomaly proxy, không tự động vi phạm physics.** Threshold phụ thuộc FPS, smoothing, coordinate frame và reconstruction noise. Derivative bậc ba khuếch đại noise; resampling 5 → 16 FPS có thể đổi score mạnh.
2. **Link-length consistency không chứng minh đúng robot.** Một link sai kích thước nhưng giữ nguyên suốt video có thể pass. Muốn “known body” đúng nghĩa cần nominal morphology hoặc conditioning-frame reference với uncertainty. Nếu chỉ đo temporal constancy, đặt tên/claim đúng mức đó.
3. **Monocular lifting cần giải thích scale.** Nếu không có metric calibration, mm/mm³-related units không tự nhiên suy ra từ video. Ratio theo calibration distribution có ích, nhưng không biến raw depth scale thành ground truth. Với BodyJerk, đơn vị là mm/s³.
4. **Dense fidelity khác physical validity.** Dense sai và accelerated giống dense vẫn fidelity cao. Cả hai frozen có thể flow error thấp. Cần GT/control để diễn giải motion retention.
5. **Flow EPE tại cùng pixel không đảm bảo cùng vật thể.** Một dịch chuyển appearance nhỏ có thể đổi field mạnh; report foreground/object tracking diagnostics và alignment assumptions.
6. **No generative VLM judge không có nghĩa evaluator không có learned prior.** SAM, tracker, depth, optical flow, interpolation có failure modes riêng. Novelty nằm ở kiểm tra rõ ràng và validated measurement, không ở lời hứa “algorithm nên đúng”.

## 7. Bản viết contribution đề xuất

Đoạn dưới tránh số chưa reconcile và claim chưa validate. Các câu về kết quả validation/Pareto chỉ bổ sung sau khi có thí nghiệm.

> **A structured benchmark for robotic video world models.** We introduce [BenchName], a benchmark for observable structural and spatiotemporal fidelity in generated robotic rollouts. It covers three embodiment categories, single- and multi-view observations, and atomic and sequential task subsets, with explicit reporting of model coverage, conditioning modality, and adaptation protocol.
>
> **Property-specific diagnostics of body and cross-view consistency.** We assess temporal consistency of robot structure within each view and geometric and temporal agreement across generated views of the same episode. These checks produce localized evidence of inconsistency without a generative vision-language judge. We report measurement coverage and separate these structural diagnostics from motion proxies and fidelity to a dense reference.
>
> **An evaluation of acceleration-induced fidelity changes.** We evaluate nine inference accelerators across their supported model configurations using paired dense and accelerated rollouts. The protocol separates absolute consistency violations from deviation relative to dense generation, enabling property-specific analysis of the speed–fidelity trade-off.

Nếu validation xong, contribution 2 nên thêm **một con số held-out human alignment/false-positive rate**, contribution 3 thêm **một finding Pareto/ranking cụ thể**. Hai con số này có trọng lượng hơn việc nhét toàn bộ corpus counts vào Introduction.

## 8. Story gọn nên chốt

P1: Generated robot rollouts được dùng cho quyết định → visual plausibility chưa đủ.

P2: Existing evaluation có perceptual, judge-based và grounded approaches → thiếu bằng chứng đầy đủ cho body persistence và simultaneous cross-camera agreement trong setting bài.

P3: Known embodiment/camera layout cho phép operationalize hai kiểm tra này → nêu observability và perception uncertainty.

P4: [BenchName], Body/View diagnostics, validation và coverage → đưa metric names/figure ở đây.

P5: Acceleration là controlled stress test → đo physics-related failures và dense divergence riêng → nêu finding thật.

P6: Ba contributions như trên.

**Việc nên làm ngay:** reconcile manifest và conditioning; khôi phục evaluator source; validate Body/View bằng human labels độc lập + targeted controls; sau đó chạy matched acceleration subset và Pareto. Chưa cần accelerator thứ mười. Tránh tên quá gần PhyWorldBench đã có trong literature; chốt scope và evidence trước, tên sau.
