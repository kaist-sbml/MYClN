# GitHub 코드 + Hugging Face 모델 배포

GitHub에는 코드·문서·작은 예제·검증 코드를 올리고, Hugging Face의 **Model 저장소 하나**에
학습된 가중치와 kNN 검색 데이터를 함께 올립니다. 모델 저장소 루트는 다음 구조입니다.

```text
README.md
SHA256SUMS
expression/
  manifest.json
  heads/seed_*.pt
  keys_{full,dual,raw,gio}.npy
  expression.npy
  has_expression.npy
  train_mean.npy
  media_vectors.npy
  feature_stats.npz
  transforms.npz
  lightgbm.txt
  stack.npz
  reference_genes.tsv
  ...
glm/
  v7_cisfull_bgcoff_best.pth
  v5_emb_stats.npz
```

현재 모델은 약 9.7GB이고 가장 큰 파일은 약 4.1GB입니다.
Hugging Face는 대형 모델 파일 업로드를 지원하므로 이 파일들을 tar로 묶거나 수동 분할할 필요가 없습니다.
계정의 저장 공간 한도는 공개 범위와 요금제에 따라 확인하세요.
[공식 저장 한도](https://huggingface.co/docs/hub/storage-limits),
[공식 업로드 안내](https://huggingface.co/docs/huggingface_hub/guides/upload)

가중치·검색 인덱스·정규화 통계·배지별 발현량 테이블은 같은 학습 실행에서 나온 한 묶음입니다.
따로 갱신하지 말고 같은 버전으로 배포합니다. 원본 학습 데이터 자체를 별도로 공개할 때는
추론 모델과 구분한 Dataset 저장소를 추가할 수 있습니다.

## 게시자가 직접 실행할 명령

Hugging Face 웹에서 **New Model**을 선택해 `OWNER/MODEL_REPO`를 만들고 공개 범위를 정합니다.
아래 명령의 `OWNER/MODEL_REPO`를 실제 저장소 ID로 바꾸세요.
모든 명령은 로컬 소스 저장소 루트에서 실행합니다.

```bash
python -m pip install -U huggingface_hub
hf auth login

# 로컬 파일 완전성 확인, SHA256SUMS 및 모델 카드 준비. 업로드는 하지 않음.
python scripts/prepare_hf_metadata.py
```

`artifacts/README.md`에 GitHub 코드 주소와 적용할 모델/데이터 라이선스를 채웁니다.
모델 카드 원본 템플릿은 `docs/HF_MODEL_CARD.md`이며 준비 스크립트는 이미 있는 카드를 덮어쓰지 않습니다.
이번 작업 폴더에는 카드와 체크섬을 준비해뒀습니다. 모델 파일을 변경했다면 준비 스크립트를 다시 실행합니다.

```bash
hf upload OWNER/MODEL_REPO ./artifacts . --repo-type model \
  --include "expression/**" "glm/**" "README.md" "SHA256SUMS"
```

파일 선택은 발현 모델과 gLM에 한정합니다. `artifacts/reference/` 같은 학습 원본, 로컬 설정,
코드, 다운로드 캐시는 올라가지 않습니다. tar 파일도 업로드할 필요가 없습니다.
CLI는 로그인 시 저장한 자격 증명을 사용합니다. 명령이나 문서에 토큰을 직접 쓰지 않습니다.

모든 파일이 올라간 최종 상태에 `v1.0.0` 같은 모델 태그를 만들고, GitHub 코드 릴리스에서
그 모델 태그 또는 commit SHA를 명시하세요. 양쪽 저장소에 서로의 링크를 추가합니다.
현재 템플릿에 있는 ID와 버전은 예시이며 실제 저장소나 태그가 만들어졌다는 뜻이 아닙니다.

## 사용자가 내려받는 명령

GitHub 소스를 clone하고 의존성을 설치한 뒤, 그 폴더에서 실행합니다.

```bash
python -m pip install -U huggingface_hub
hf download OWNER/MODEL_REPO --local-dir artifacts
sha256sum -c artifacts/SHA256SUMS
python -c "from model_artifacts import check_model; print(check_model('artifacts/expression'))"
```

다운로드 결과가 코드의 기본 `artifacts/expression/`, `artifacts/glm/` 경로와 바로 일치합니다.
무결성 검사가 통과하면 README의 입력 생성·예측 명령을 실행합니다. 학습은 필요 없습니다.
비공개 또는 접근 제한 모델에는 권한이 있는 계정으로 `hf auth login`이 필요합니다.

재현용 버전 고정은 다음과 같습니다. 실제 게시된 태그나 commit SHA를 지정하세요.

```bash
hf download OWNER/MODEL_REPO --revision v1.0.0 --local-dir artifacts
```

`--local-dir`는 다운로드 메타데이터를 보관해 이후 변경된 파일을 받는 데 사용합니다.
[공식 CLI 안내](https://huggingface.co/docs/huggingface_hub/guides/cli),
[공식 다운로드 안내](https://huggingface.co/docs/huggingface_hub/guides/download)

GitHub Release 분할 파일 방식은 기존 대안으로 남겨두었으며, 이 구성에서는 실행하지 않습니다.
