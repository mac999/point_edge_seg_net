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
