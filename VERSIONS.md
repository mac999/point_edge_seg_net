# Version History

실험 버전별 코드/설정/결과 추적. 각 버전은 git tag와 1:1 대응.
평가는 전부 **full protocol** (Area_5 held-out, chunk mode, grid sampler, halo 1.0, core만 채점).

| version | tag | mIoU | OA | params | 핵심 변경 |
|---|---|---:|---:|---:|---|
| 0.5 | (upstream) | — | — | 5.7M | GitHub 공개 상태 (2025/10/1). block 8192, FPS sampler |
| 0.6.0 | `v0.6.0` | **58.92** | 84.81 | 5.73M | 현재 baseline (logs/20260730_121057). voxel-chunk 파이프라인(4cm voxel + KD median split + halo 1.0 + 20480 블록), grid sampler, strong augmentation, Focal+Lovász, train/eval 피처 경로 단일화 |
| 0.6.1 | `v0.6.1` | 58.82 | 84.70 | **3.15M** | REBAL2 (logs/20260813_082752 + 20260814_233522): enc (64,192,320,448) + bottleneck 256. mIoU 동률로 파라미터 45% 절감 → 신규 기본 아키텍처. 병목 비중 74.5%→41.3% |

## v0.6.1 실험 노트 (2026-08-14~15)

- **환경 교체**: 공용 `dl` env가 torch 2.13(LLM 스택)으로 넘어가며 PyG 소실 →
  전용 conda env **`pesn`** 신설 (torch 2.7.1+cu128, torch_cluster/scatter/sparse
  pt27cu128 프리빌트 휠, PyG 2.8.0). Blackwell sm_120 커널 동작 실측 확인.
- **결과**: capacity 재배분으로 정확도 개선 실패(−0.10, 노이즈), 파라미터 45% 절감 성공.
  진단(전 해상도 conv1 starvation)의 핵심 변인 c1=64는 미변경 → 가설 미검증 상태.
- **모델 선택 지표 불신 증거**: ep134(val_miou 81.51) test 57.93 < ep145(80.89) test 58.82.
  val 지표와 test가 역전 → best_model.pth만 믿지 말고 final도 함께 평가할 것.
- **성능 프로파일 (N=98,304, 방 단위 크기)**: forward의 **99%가 knn_graph**.
  스테이지당 동일 그래프 2회 중복 계산(conv_n/conv_n_2) → 그래프 재사용 시 ~40% 단축 가능.
- **VRAM 스케일링 실측**: fwd+bwd allocated ~105MB/1k점 (선형). 최대 방(291k voxel) 32GB.
  단, 가변 크기 입력의 단편화로 reserved는 allocated의 1.5~3배까지 관측됨 →
  방 단위 학습 시 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + 크기 버킷팅 필요.

## v0.8.x 실험 로그 (진행 중, 2026-08-15)

**목표**: model_v2.py (`PointEdgeSegNetV2`, 직렬화 이웃 + Meta 집계)로 v1 품질(58.82)을
유지하면서 속도·메모리 구조 개선을 확정한다. 브랜드명은 PointEdgeSegNet 유지.

| # | 실험 | 설정 | val mIoU | test mIoU | 판정 |
|---|---|---|---:|---:|---|
| AB | 직렬화 1차 | k=32, 곡선1, 150ep | 63.7@ep148 (상승 중) | **47.62** | v1 대비 −11.2. 속도 ×7.7 (63분/150ep), VRAM ÷3 (평균 10.8GB) 확정 |
| E1 | 윈도우 확대 | k=64, 곡선1, ep142 조기종료 | 62.4 | **49.34** | +1.7 — 폭 확대는 부분 회복만 (sofa +10.6, door +6.8) |
| E2 | 곡선 병용 | k=64, 곡선2, 150ep | 63.6 | **49.35** | E1과 동률 — 곡선 수는 k=64에선 무차별 (sofa만 +15.0으로 최다 회복) |

**E1/E2 판정 (2026-08-15 14:11)**: 이웃 폭·이음새 가설은 11.2p 격차의 **1.7p만 설명**.
door(23~29 vs v1 48)·board(34~36 vs 60)·chair(62~64 vs 77)는 폭을 2배로 늘려도 회복
불가 → 잔여 격차의 주범은 **직렬화 이웃의 근사 품질 자체**. 핵심 통찰: SOTA 조합을
반쪽씩 가져온 것이 패인 — PTv3는 근사 직렬화 이웃이지만 **1024-패치 어텐션**으로 보정
하고, DeLA는 단순 max 집계지만 **정확한 kNN**을 쓴다. 우리는 "근사 이웃 + 단순 집계"
라는, 아무도 출시하지 않은 약한 반쪽 2개의 조합을 만들었다. 세 런 모두 train acc
82~84%에서 종료(언더핏)라 장기 학습 여지 +2~4p 별도 존재.

**AB에서 배운 것**:
- 클래스 패턴이 원인을 지목: sofa −39.5, door −26.2, board −24.2, chair −12.7 (소형·
  얇은 구조물 붕괴) vs floor/ceiling/window 유지 → 직렬화 ±16 윈도우가 kNN-32보다
  물리적으로 좁다는 시그니처.
- 명백한 언더핏: train acc 84%에서 150ep 종료 (v1은 96% 수렴). v2는 같은 에폭 예산으로
  부족 — 에폭 23초이므로 600ep(±4h) 장기 런이 다음 카드.
- 툴링 버그 2건 기록: ① run_v2_variants.sh의 `P=$(launch ...)`는 서브셸에서 bg 실행
  → 부모의 `wait`가 exit 127. 채점은 run_v2_score_variants.sh로 재부착(PID 폴링).
  ② 균일 랜덤 점으로 v2를 벤치마크하면 grid pooling이 병합을 못 해 병목 어텐션이
  전점에 걸림(51GB 폭발) — v2 벤치마크는 반드시 실데이터(표면형)로.

| E3 | v2.1 stencil r1 | K=27+diff, ep136 조기종료 | 79.0 | **59.32** | **v1 추월** (+0.5) — 실효 이웃 9개로 |
| E4 | v2.1 stencil r2 | K=125+diff, ep132 조기종료 | 81.4 | **62.03** | **신기록 +3.21** — 프로젝트 사상 최고 |

**E3/E4 판정 (2026-08-15 21:41) — v2.1 성공, 새 baseline = E4 62.03**:
- 근사이웃 주범 가설 확증: 정확한 stencil 이웃으로 바꾸자 serial 대비 +12.7 (49.35→62.03).
- v1 대비 클래스별: door +9.5, window +8.3, chair +8.1, sofa +6.4, bookcase +6.0 —
  serial에서 붕괴했던 클래스들이 회복을 넘어 v1을 크게 추월. beam은 여전히 0.
- E4가 E3보다 +2.7 → 이웃 폭도 유효 (r2 ~25개 ≈ kNN-32급이 정답).
- best(60.78) < final(62.03) 재확인 — val 선택 지표 불신 3번째 증거.
- 속도: E4 학습 6h(병렬 공유·300W 캡 하) — 단독이면 ~2.5h. VRAM 피크 ~50GB(병렬 합).
- 결론: PointEdgeSegNetV2(stencil r2, feature_diff)가 다음 릴리스의 PointEdgeSegNet
  본체 후보. 2.54M params로 v1(3.15M) 대비 −19% 파라미터, +3.2 mIoU.

**v2.1 구현 노트 (2026-08-15 오후)**: grid-stencil 이웃(`stencil_neighbors`) +
feature-diff 항(`MetaBlock(feature_diff=True)`, W₂(h_j−h_i)를 점별 행렬곱+gather로 분해).
성능 버그 2건을 잡음 — ① 오프셋별 파이썬 루프(136ms)를 모튼 N×K 일괄 + searchsorted
1회로 벡터화(1.3ms). ② 빈 셀 sentinel=0이 backward scatter_add를 0번 행 한 주소에
직렬화(6.5s/step) → sentinel을 각 점의 자기 인덱스로 분산(45ms/step, 결과 불변).
최종: r1 45ms/1.1GB, r2 170ms/4.0GB (fwd+bwd, batch4). CLI: --v2_neighbors stencil
--v2_stencil {1,2} --v2_diff.

**다음 후보 (v2.1 설계)**: ① **grid-stencil 이웃** — 모든 스테이지의 점이 이미 복셀
격자 위에 있으므로(4cm 입력 + grid pooling), 이웃 = 고정 복셀 오프셋의 해시 조회
(정렬 1회 + searchsorted). 탐색 0회에 **정확한 공간 이웃** — sparse conv(MinkowskiNet)
의 kernel map과 동일 원리로, 직렬화의 속도와 kNN의 정확성을 동시에 취함.
② MetaBlock에 **feature-difference 항** 복원: e = W₁h_j + W₂(h_j−h_i) + posenc —
EdgeConv의 기하 그래디언트 큐를 점별 행렬곱+gather 분해로 재현(엣지 MLP 없이).
③ 승자 구성 + **600ep 장기 런** (언더핏 해소, 에폭 23~40초라 저렴).
④ num_workers>0 (I/O 병목, util 75%).

## v0.8.x 후속 실험 큐 (2026-08-15 밤 가동)

`run_v08_queue.sh` — **재시작 가능**: 크래시/재부팅 시
`cd ~/project/point_edge_seg_net && nohup ./run_v08_queue.sh >> v08_queue.log 2>&1 &`
로 다시 실행하면 `logs/v08_queue.state`를 읽어 이어서 진행 (완료 단계 skip, 중단 학습은
checkpoint에서 --resume, 채점은 결과 JSON 존재로 skip 판정).

**Q1 E5 결과 (2026-08-16 17:01)**: ep517 조기종료 (val best 84.60@ep499, train 97.8%).
full-protocol **best 64.11 / final 64.23 — 신기록 +2.20** (62.03→64.23). OA 87.23,
mAcc 72.31. 클래스별: door +9.8(67.2), column +5.5, clutter +4.6, table +3.8,
board +3.6, floor +2.2 / window −5.8(유일한 큰 후퇴), sofa −0.8. beam 여전히 0.
final>best 4번째 확인. MinkowskiNet(65.4)까지 1.2p.

| 단계 | 내용 | 예상 |
|---|---|---|
| Q1 E5 | E4 레시피(stencil r2+diff) × **600ep** — 언더핏 해소 | ✅ **64.23** |
| Q2 | E5·E4 final에 **TTA 10뷰** 채점 (5스케일×플립) | ~30분 |
| Q3 E6 | **room 모드**(방 전체 복셀) × 300ep — v1 시절 실패했던 경로를 v2.1로 재도전 | ~4-6h |
| Q4 E7 | **2cm 복셀** 캐시 신규(core 24576/block 49152) × 150ep, base_grid 0.02 | ~10h+ |

**운영 노트 (2026-08-16)**: 장기 런에서 nvidia-smi VRAM이 14→80GB 톱니파를 그림 —
누수가 아니라 캐싱 얼로케이터의 가변 크기 재사용 실패 + 100배치마다 empty_cache 반납
주기임 (train_model.py:1157). allocated는 4-8GB 수준. 새 런에는
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 붙일 것.

SOTA 위치 (2026-08-15): full-protocol 62.03은 RandLA-Net(~62.4)·GACNet(62.9)·
HPEIN(61.9)과 동급 = **2019-2020 point-based 중위 티어**. 하루 만에 PointCNN(2018)
티어에서 두 세대 상승. 다음 벽: MinkowskiNet 65.4 → KPConv 67.1.

## 로드맵 (계획)

- **0.7.x — 컨텍스트 윈도우 확장**: (A) core_max 49152 청크 재생성 실험 →
  (B) 방 전체 학습(room-as-block, 크기 버킷팅). 사전 작업: knn_graph 스테이지 재사용.
- **0.8.x — capacity 진단 완결**: c1 64→128/192 (전 해상도 단계 확대).
- 후보: 2-스트림 coarse 컨텍스트(cross-attention, zero-init), DeLA식 분리 집계,
  장기 학습(600ep+), TTA/투표 프로토콜 적용.

## 운영 규칙

1. 실험 하나 = run 스크립트 하나(`run_*.sh`) + 설정 JSON. 결과가 유의미하면
   VERSIONS.md에 행 추가 후 커밋, `git tag -a v0.x.y`.
2. `logs/`, 데이터/캐시, `*.pth`는 git 밖 (.gitignore). 수치는 이 파일에 기록.
3. 아키텍처 인자(enc_channels, bottleneck_dim, sampler, block 파라미터)는
   체크포인트와 1:1 — 평가 시 학습과 동일 인자 필수.
