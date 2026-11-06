# Actinomycete gene-expression model

방선균 유전체와 배지 정보를 이용해 유전자별 발현량을 예측하는 최종 모델의 소스 배포판입니다.
기존 `Final_model/release_ver`의 **BGC-free gLM + media** 구성을 기준으로 정리했습니다.

모델은 ESM2-35M 단백질 임베딩, gLM 유전체 문맥, DNA 조성, 배지 벡터를 사용합니다.
학습 유전자의 발현량을 kNN으로 검색한 뒤 8개 seed의 FiLM head, LightGBM, ridge stacking으로 예측합니다.
**학습은 최초 1회 수행해 저장하고, 예측 명령은 저장된 모델만 불러옵니다.**
gLM 체크포인트는 `v7_cisfull_bgcoff_best.pth`이며 cis encoder 설정은 96/4입니다.

## 빠른 시작: 저장 모델로 예측

Python 3.10 환경에서 검증했습니다. 소스와 학습된 모델은 별도로 배포합니다.
**학습된 모델 번들을 받았다면 `train.py`를 실행할 필요가 없습니다.**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-preprocessing.txt
```

PyTorch는 사용 장비와 호환되는 CPU/CUDA wheel이 필요합니다.
`requirements.txt`는 허용 범위, `requirements-tested.txt`는 A100 검증 환경의 직접 의존성 버전입니다.

모델은 Hugging Face에서 저장소의 `artifacts/` 폴더로 내려받는 구성을 사용합니다.
`OWNER/MODEL_REPO`는 배포자가 게시한 실제 모델 저장소 ID로 바꾸세요.
현재 이 표기는 예시이며 업로드된 저장소 주소를 뜻하지 않습니다.

```bash
python -m pip install -U huggingface_hub
hf download OWNER/MODEL_REPO --local-dir artifacts
sha256sum -c artifacts/SHA256SUMS
python -c "from model_artifacts import check_model; print(check_model('artifacts/expression'))"
```

게시된 모델 태그나 commit SHA를 `--revision`으로 지정하면 버전을 고정할 수 있습니다.
기존 tar 번들을 받은 경우에는 `tar -xf /path/to/Final_release-model.tar`로 풀어도 됩니다.
[Hugging Face 업로드·다운로드 안내](docs/HUGGINGFACE.md)를 참고하세요.

`artifacts/expression/`에 검색 인덱스·head·LightGBM·ridge·정규화 통계가,
`artifacts/glm/`에 gLM 가중치와 입력 정규화 통계가 놓입니다.
ESM2-35M은 첫 입력 생성 시 공식 가중치를 다운로드해 캐시에 보관합니다.
기존 로컬 작업 폴더에는 이 모델들이 이미 준비되어 있습니다.

예측에는 원본 학습 pickle이나 taxonomy CSV가 필요하지 않습니다. 기본 폴더에 모델을 풀었다면
`config.local.json`도 만들 필요가 없습니다. 다른 위치의 모델은 `--model-dir`로 지정합니다.

학습을 재현할 경우에만 reference 파일 경로를 `config.local.json`에 설정하고
[학습·추론 안내](docs/SERVING.md)를 따르세요. JSON의 상대 경로는 JSON 파일이 있는 폴더 기준이며,
`ACTEXPR_<설정명>` 환경변수로 개별 경로를 덮어쓸 수 있습니다.

## 원시 유전체에서 입력 생성·예측

GenBank 또는 nucleotide FASTA와 배지 이름/조성을 넣으면 필요한 임베딩과 입력 파일을 자동 생성합니다.

```bash
python -m pip install -r requirements-preprocessing.txt
python prepare_inputs.py --genbank genome.gbk \
  --medium ISP2 --out outputs/my_genome --device cuda
```

FASTA는 `--fasta genome.fna --topology linear`로 지정하면 CDS 호출부터 실행합니다.
배지 조성 JSON은 `--medium ISP2` 대신 `--medium-json examples/medium_composition.json`으로 넣습니다.
전체 흐름은 **CDS → ESM2-35M → UniOP operon → promoter/topology → gLM → query.npz**입니다.
동일한 폴더에 `genome.json`, `medium.json`도 생성됩니다.

```bash
python examples/predict_json.py \
  --model-dir artifacts/expression --device cuda \
  --genome outputs/my_genome/genome.json \
  --medium outputs/my_genome/medium.json \
  --embeddings outputs/my_genome/query.npz \
  --output outputs/my_genome/predictions.json
```

처음부터 `prepare_inputs.py`에 `--predict`를 추가하면 입력 생성과 발현 예측을 한 번에 실행합니다.
이때도 미리 학습된 발현 모델이 필요하며 `--model-dir`로 지정할 수 있습니다.
입력 생성에는 gLM 가중치·정규화 통계·ESM 모델이 필요합니다. 발현 예측에는 저장된 발현 모델과
query 파일만 필요하고, 원본 학습 pickle·taxonomy 파일을 읽지 않습니다.
모델이 없거나 저장이 중단된 상태면 즉시 종료하며 자동으로 학습하지 않습니다.

[원시 입력 사용법과 실행 예제](docs/RAW_INPUTS.md)에 설치, FASTA gene calling,
배지 인코딩, 오프라인 가중치 사용법을 정리했습니다.
이미 계산된 임베딩을 직접 전달할 때의 계약은 [docs/INPUTS.md](docs/INPUTS.md)를 참고하세요.

## 구성

| 경로 | 내용 |
|---|---|
| `expression_model.py` | retrieval, FiLM head, LightGBM, ridge 모델 |
| `train.py` | 최초 1회 학습·평가 및 모델 저장 |
| `predict.py`, `serving.py`, `model_artifacts.py` | 저장 모델 검증·로드 및 새 유전체 추론 |
| `prepare_inputs.py` | GenBank/FASTA + 배지 → 전체 추론 입력 자동 생성 |
| `config.py`, `configs/head.json` | 포터블 경로 설정과 최종 튜닝값 |
| `schema.py`, `data.py`, `split.py` | 데이터 계약, 변환, genus 단위 분할 |
| `glm/` | 최종 gLM 아키텍처, 사전학습, 임베딩 추출 |
| `preprocessing/` | CDS 호출·ESM·UniOP·promoter·topology·배지 인코딩 및 참조 데이터 병합 |
| `results/reference/` | 기존 최종 실행의 결과 JSON과 신뢰구간 |
| `tests/`, `.github/workflows/ci.yml` | CPU 통합 검증 및 GitHub CI |
| `artifacts/glm/` | 현재 폴더에 별도 보관한 gLM 가중치와 정규화 통계; Git 제외 |
| `artifacts/expression/` | `train.py` 완료 후 생성되는 발현 모델; Git 제외 |

## 성능과 재현 범위

기존 최종 실행의 test Pearson r은 **0.637364**, within-condition r은 **0.643334**입니다.
22개 test 균주를 단위로 한 bootstrap 95% 신뢰구간은 **[0.619063, 0.657530]**입니다.
저장된 실제 split 목록은 **133 train / 5 validation / 22 test = 160개**입니다.
기존 문서의 “161-strain”은 실험 식별용 표기이며 실제 결과 목록의 합과 다릅니다.
이 결과는 6개 genus를 함께 hold out한 canonical 평가이며 LOGO 평균이 아닙니다.

이후 학습·저장을 완료한 배포 모델의 test Pearson r은 **0.637365**, within-condition r은
**0.643334**입니다. 위 신뢰구간은 기존 실행에 대한 값이며 새 실행에서 다시 계산하지 않았습니다.
저장된 모델을 새 프로세스에서 로드한 실제 유전체 추론도 확인했습니다.
검증 이력은 [docs/VALIDATION.md](docs/VALIDATION.md)를 참고하세요.

S. venezuelae ATCC 10712 / ISP2의 7,132개 유전자에 대해 A100 80GB에서 측정한 1회 실행은
입력 생성 **46.1초**, 모델 로드·종료를 포함한 추론 **10.2초**, 합계 **56.3초**였습니다.
추론 함수 내부 시간은 4.1초입니다. 가중치와 OS 파일 캐시가 준비된 환경의 측정이며
일반 GPU의 속도나 최소 VRAM을 보장하는 결과는 아닙니다. [자원 측정](docs/RESOURCE_USAGE.md)을 참고하세요.

## GitHub 업로드

코드는 [GitHub 업로드 안내](docs/GITHUB_UPLOAD.md), 모델은 [Hugging Face 배포 안내](docs/HUGGINGFACE.md)를 따릅니다.
코드, 설정 예시, 문서, 작은 평가 JSON을 Git에 올리면 됩니다.
`.gitignore`는 로컬 설정, 데이터, 모델 가중치, 실행 결과, 캐시를 제외합니다.
별도 소스 ZIP은 다음 명령으로 생성합니다.

```bash
python scripts/package_release.py
# dist/Final_release-source.zip
```

ZIP에는 서버 경로, 로컬 설정, 모델 가중치, 원시 데이터가 포함되지 않습니다.
## 모델 배포

학습된 발현 모델·kNN 검색 데이터·gLM 가중치는 Hugging Face Model 저장소 하나에 함께 배포합니다.
`artifacts/expression/`과 `artifacts/glm/`을 그대로 올리므로 tar 분할이 필요 없습니다.
GitHub에는 코드만 올리고, 코드 버전에 맞는 Hugging Face 모델 태그/commit을 문서에 명시합니다.

```bash
python scripts/prepare_hf_metadata.py
# artifacts/README.md 및 artifacts/SHA256SUMS 준비; 업로드는 수행하지 않음
```

오프라인 전달용 tar가 필요한 경우에는 다음 기존 명령을 사용할 수 있습니다.

```bash
python scripts/package_model.py
# dist/Final_release-model.tar + dist/Final_release-model.SHA256SUMS
```

번들은 `artifacts/expression/`(학습셋 133개 균주 기준 kNN 검색 인덱스·발현 테이블·`reference_genes.tsv`,
FiLM head 8개, LightGBM, ridge, 정규화 통계, 평가 결과)와 `artifacts/glm/`을 포함합니다.
소스 checkout 루트에서 풀면 바로 `examples/predict_json.py`로 예측할 수 있습니다.
원시 유전체 입력 생성용 gLM/ESM 가중치도 필요합니다. [아티팩트 목록](docs/ARTIFACTS.md)을 참고하세요.
소스의 기존 권리·라이선스는 이 정리에서 변경하지 않았습니다.
