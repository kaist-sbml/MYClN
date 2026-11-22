# MYClN: 한 번 학습하고 저장 모델로 예측하기

ActinoBERT는 genomic backbone, MYClN은 전체 발현 예측 모델의 이름입니다.

`train.py`가 학습과 모델 저장을 담당하고 `examples/predict_json.py`는 추론만 합니다.
예측 중에는 reference 전체의 kNN 재계산, FiLM 학습, LightGBM 학습, ridge fitting이 없습니다.

## 1. 저장 모델 준비

기본 배포는 Hugging Face Model 저장소입니다. 저장소 루트에서 모델을 받으면 기본 경로에 놓입니다.
`OWNER/MODEL_REPO`를 실제 게시된 저장소 ID로 바꾸세요.

```bash
python -m pip install -U huggingface_hub
hf download OWNER/MODEL_REPO --local-dir artifacts
sha256sum -c artifacts/SHA256SUMS
python -c "from model_artifacts import check_model; print(check_model('artifacts/expression'))"
```

자세한 게시·다운로드 방법과 버전 고정은 [HUGGINGFACE.md](HUGGINGFACE.md)에 있습니다.
기존 tar를 받았다면 `tar -xf /path/to/Final_release-model.tar`로 복원한 후 같은 검사를 합니다.
재학습은 필요 없으며 준비가 끝났으면 아래 2번 입력 준비로 이동합니다.

### 학습 재현이 필요한 경우에만

저장소 루트에서 실행합니다. 기존 reference 파일 경로를 설정한 `config.local.json`이 필요합니다.

```bash
export ACTEXPR_CONFIG="$PWD/config.local.json"
export ACTEXPR_DEVICE=cuda
python train.py --check
python train.py --output artifacts/expression
```

전체 참조 데이터에 대한 이 작업은 오래 걸립니다. reference 로드, 네 가지 kNN 인덱스,
8개 seed head, LightGBM/ridge 순서로 진행 상황을 출력합니다. gLM을 다시 사전학습하지 않고
이미 계산된 reference ESM/gLM 임베딩으로 발현 모델을 학습합니다.

완료 시 `Model ready`가 출력되고 모델 폴더에 `manifest.json`이 생성됩니다.
중단한 학습은 완성 모델로 취급하지 않습니다. 같은 폴더에서 처음부터 재학습하려면
`--overwrite`를 지정합니다. 중간 epoch에서 이어 학습하는 기능은 없습니다.

| 저장 파일 | 역할 |
|---|---|
| `keys_{full,dual,raw,gio}.npy` | 정규화된 training gene 검색 인덱스 |
| `train_mean.npy`, `expression.npy`, `has_expression.npy` | training gene 평균 및 배지별 발현량 |
| `media_vectors.npy` | 정확한 배지 벡터 조회 |
| `reference_genes.tsv` | 검색 인덱스 행 → (genome_id, gene_id) 매핑 |
| `feature_stats.npz`, `transforms.npz` | 학습 시 고정한 정규화 통계 |
| `heads/seed_*.pt` | 검증 기준으로 선택한 FiLM 가중치 |
| `lightgbm.txt`, `stack.npz` | LightGBM 및 ridge 가중치·정규화 |
| `reference_metadata.json`, `fit_metadata.json` | 인덱스 정보, head 구조, k/temperature |
| `manifest.json` | 저장 형식·완료 표시·필수 파일 크기 |
| `result.json`, `test_predictions.npz` | 해당 실행의 평가 결과 |

모델 배포 시 폴더 전체를 함께 전달하세요. 모델에는 training reference에서 유래한
검색용 특징과 발현량 테이블이 포함됩니다. 원본 참조 pickle·taxonomy CSV는 추론에 필요하지 않습니다.
manifest 검사는 누락·잘린 파일을 확인하는 용도이며 같은 크기의 내용 변경을 탐지하는 해시 검사는 아닙니다.

### 모델 배포 번들

```bash
python scripts/package_model.py
# dist/Final_release-model.tar, dist/Final_release-model.SHA256SUMS
```

번들에는 `artifacts/expression/` 전체(검색 인덱스, 발현 테이블, FiLM head, LightGBM, ridge,
정규화 통계, `reference_genes.tsv`, 평가 결과)와 `artifacts/glm/`의 gLM 가중치·정규화 통계가 들어갑니다.
소스 checkout 루트에서 풀면 기본 경로에 그대로 놓이며 `sha256sum -c artifacts/SHA256SUMS`로 검증합니다.
`reference_genes.tsv`가 없는 이전 export에는 `python scripts/export_reference_genes.py`로 추가합니다.

## 2. 입력 준비

```bash
python prepare_inputs.py --genbank genome.gbk --medium ISP2 \
  --out outputs/my_genome --device cuda
```

이미 `genome.json`, `medium.json`, `query.npz`를 생성했다면 그대로 재사용합니다.
입력 생성에는 ESM2-35M과 최종 BGC-free gLM 가중치가 필요합니다.

## 3. 예측

```bash
python examples/predict_json.py \
  --model-dir artifacts/expression --device cuda \
  --genome outputs/my_genome/genome.json \
  --medium outputs/my_genome/medium.json \
  --embeddings outputs/my_genome/query.npz \
  --output outputs/my_genome/predictions.json
```

모델이 없으면 입력 파일을 읽기 전에 종료하며 자동 학습하지 않습니다.
`--model-dir`를 생략하면 `config.MODEL_DIR`을 사용합니다. 기본은 저장소의 `artifacts/expression`입니다.
다른 위치에 저장했다면 `--model-dir` 또는 `ACTEXPR_MODEL_DIR`로 지정하세요.
출력은 `{gene_id: predicted_expression}` JSON이며 기존 출력은 덮어쓰지 않습니다.

추론도 query × reference 유사도 계산이 필요하므로 시간이 0이 되는 것은 아닙니다.
GPU에는 reference key를 타일 단위로 전송하고 전체 인덱스는 CPU에서 memory-map으로 읽습니다.
첫 실행에는 모델 파일을 디스크에서 읽는 시간이 추가될 수 있습니다. A100 80GB에서의
실제 유전체 1회 실행 시간과 메모리는 [RESOURCE_USAGE.md](RESOURCE_USAGE.md)에 기록했습니다.
CPU는 `--device cpu --threads 4`를 사용합니다.
CUDA와 CPU, 타일 크기에 따른 부동소수점 차이는 있을 수 있습니다.

Python에서 여러 번 사용할 때는 모델을 한 번 로드해 재사용할 수 있습니다.

```python
from serving import ExpressionPredictor
model = ExpressionPredictor("artifacts/expression", device="cuda")
result = model.predict(genome, medium, embeddings)
```

학습에 없던 배지 벡터에는 정확한 배지 일치에 의한 발현량 보정이 적용되지 않습니다.
이 경우에도 training gene 평균 기반 retrieval과 배지 입력을 이용한 head/stack 예측은 수행합니다.
