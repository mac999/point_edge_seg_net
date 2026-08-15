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
| E1 | 윈도우 확대 | k=64, 곡선1, 150ep | (진행 중, logs/20260815_110116) | — | 이웃 폭 가설 검증 |
| E2 | 곡선 병용 | k=64, 곡선2, 150ep | (진행 중, logs/20260815_110121) | — | 이음새 가설 검증 |

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

**다음 후보** (E1/E2 결과에 따라): 승자+600ep 장기 런 / 하이브리드(stage1-2 직렬화 +
stage3-4 정확 kNN — 깊은 스테이지는 N이 작아 kNN이 저렴) / num_workers>0 (I/O 병목,
util 75%).

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
