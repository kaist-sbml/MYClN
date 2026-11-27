# 원시 유전체에서 예측 입력 만들기

`prepare_inputs.py`가 GenBank 또는 nucleotide FASTA와 배지 정보에서 필요한 파일을 생성합니다.
수동으로 ESM·gLM 임베딩, operon NPZ, 배지 벡터를 만들 필요가 없습니다.
내부 유전체 식별자는 입력 파일에서 자동으로 생성합니다. 저장 위치는 `--out`으로 지정하면 됩니다.

## 설치

```bash
python -m pip install -r requirements-preprocessing.txt
```

ESM2-35M은 첫 실행에서 공식 가중치를 내려받아 PyTorch 캐시에 저장합니다.
오프라인에서는 `--esm-checkpoint /path/to/esm2_t12_35M_UR50D.pt`를 지정하고
같은 위치에 `esm2_t12_35M_UR50D-contact-regression.pt`도 놓으세요.
최종 gLM 가중치와 정규화 통계는 `artifacts/glm/` 기본 경로 또는
`--glm-checkpoint`, `--emb-stats`로 지정합니다. 이 두 파일은 로컬 배포에는 있고 소스 ZIP에서는 제외됩니다.

## GenBank 입력

```bash
python prepare_inputs.py \
  --genbank genome.gbk \
  --medium ISP2 \
  --out outputs/my_genome \
  --device cuda
```

CDS annotation에서 단백질·DNA 서열과 좌표를 읽습니다. 번역 서열이 없으면
`transl_table`과 `codon_start`를 반영해 번역합니다. pseudogene은 제외하고 manifest에 개수를 기록합니다.
중복 gene ID, 내부 stop codon, compound CDS 위치는 조용히 변경하지 않고 오류로 알립니다.

## 원시 FASTA 입력

```bash
python prepare_inputs.py \
  --fasta genome.fna \
  --topology linear \
  --medium ISP2 \
  --out outputs/my_genome \
  --device cuda
```

Pyrodigal로 CDS를 호출합니다. 기본은 `--gene-caller-mode meta`이며,
충분한 길이의 완성 유전체에는 `--gene-caller-mode single`을 지정할 수 있습니다.
FASTA에서는 `--topology linear` 또는 `circular`를 명시해야 합니다.
여러 contig가 서로 다른 topology를 갖는 경우 각 contig에 annotation이 있는 GenBank를 사용하세요.
새로 호출한 CDS는 기존 annotation의 CDS와 다를 수 있습니다.

Pyrodigal의 coordinate/translation API는 [공식 문서](https://pyrodigal.readthedocs.io/en/v3.5.2/api/genes.html)를 따릅니다.

## 배지 입력

알려진 배지는 `--medium ISP2`처럼 이름으로 지정합니다.
동결한 159개 성분 순서와 이름별 벡터는 `configs/media_catalog.json`에 포함했습니다.
이름별 기본 조성은 제공된 배지표의 38개 조성을 사용합니다. M9, CGXII, Media12 등도 이름으로 선택할 수 있습니다.
BHI와 HIB는 표에 조성이 명시되지 않아 `--medium-json`으로 조성 또는 벡터를 직접 입력해야 합니다.
읽기 쉬운 성분·농도 dictionary는 `configs/media_recipes.json`, 원본 표는 `configs/media_recipes.tsv`입니다.
`python scripts/build_media_catalog.py`로 원본 표에서 배포 dictionary를 재생성할 수 있습니다.
`component_aliases`에 명시된 여섯 표기만 변환하며, `SMMa`는 각주 a가 붙은 `SMM`으로 해석합니다.
수정된 기본 조성은 기존 학습 데이터·가중치·참조 배지 벡터를 변경하지 않습니다.
새 조성이 기존 참조 벡터와 일치하지 않으면 retrieval의 동일 배지 보정은 0이 됩니다.
알 수 없는 이름·성분은 오류로 알려주며 0 벡터로 바꾸지 않습니다.

직접 조성은 다음처럼 입력합니다. 값은 참조 데이터에서 사용한 수치 단위를 따릅니다.
단위 변환·동의어 통합은 자동으로 하지 않습니다.

```json
{
  "medium_id": "my_ISP2",
  "composition": {
    "Dextrose": 4.0,
    "Malt extract": 10.0,
    "Yeast extract": 4.0
  }
}
```

```bash
python prepare_inputs.py --genbank genome.gbk \
  --medium-json examples/medium_composition.json --out outputs/my_genome
```

조성 JSON의 성분은 반드시 vocabulary에 있는 정확한 이름을 사용하세요.
기존 159차원 `vector`가 있으면 `composition` 대신 넣어도 됩니다.
`params`는 기록용이며 이 최종 모델의 별도 조건 채널로 자동 인코딩되지 않습니다.

## 생성 파일

```text
outputs/my_genome/
├── genome.json                 # 유전자 ID, 서열, 좌표, promoter
├── medium.json                 # 모델 순서로 인코딩한 159차원 배지 벡터
├── query.npz                   # ESM·gLM·CLS·각 gene order
├── manifest.json               # 옵션, 입력/가중치 해시, 유전자·operon 수
└── features/<자동 생성 ID>/
    ├── cds_emb.npy
    ├── cds_meta.json
    ├── operons.tsv
    ├── promoters.fasta
    ├── operon_meta.npz
    └── topology_meta.npz
```

출력 디렉터리는 새 경로여야 합니다. 실패 시 완성되지 않은 query 폴더를 남기지 않습니다.
메모리가 부족하면 `--batch-size`를 줄이세요. CPU는 `--device cpu --threads 4`로 실행합니다.

## 발현 예측 연결

먼저 `train.py --output artifacts/expression`을 한 번 완료하거나 저장된 발현 모델 폴더를 준비합니다.
이미 준비된 query 파일은 다시 생성할 필요가 없습니다. 예측 CLI는 다음과 같습니다.

```bash
python examples/predict_json.py \
  --model-dir artifacts/expression --device cuda \
  --genome outputs/my_genome/genome.json \
  --medium outputs/my_genome/medium.json \
  --embeddings outputs/my_genome/query.npz \
  --output outputs/my_genome/predictions.json
```

또는 처음부터 `prepare_inputs.py`에 `--predict`를 추가하면 생성과 예측을 이어서 실행합니다.
입력 생성만 할 때는 대용량 expression 참조 데이터가 필요하지 않습니다.
`--predict`도 저장 모델만 불러오며 학습하지 않습니다. 모델이 없으면 ESM 계산을 시작하기 전에 종료합니다.
원본 expression 참조 데이터는 최초 모델 학습에만 필요합니다. [학습·추론 안내](SERVING.md)를 참고하세요.

## 학습 전처리와 일치시키는 규칙

- ESM: `Tools/ESM_embedding.ipynb`의 ESM2-35M, layer 12 residue mean pooling,
  비표준 아미노산을 X로 치환, 1,022 aa 초과 시 앞 511 + 뒤 511 aa.
- Operon: `Tools/Run_UniOP.ipynb`의 UniOP 방식 KDE posterior, bandwidth 0.03,
  seed 0, threshold 0.5. 외부 UniOP 패키지를 설치하는 방식은 아닙니다.
- Promoter: 전사 방향으로 upstream 250 + downstream 50 bp, contig 경계에서 자름.
  음의 strand는 reverse complement하며 원래 학습 코드처럼 원형 경계 wrap은 하지 않습니다.
- Feature: 기존 `genome_features.py`의 one-hot, 12개 promoter 수치 특징과 global 통계.
  gLM은 원본과 동일하게 promoter 수치 특징의 처음 4개를 사용합니다.
- Topology: GenBank의 linear/circular annotation을 사용합니다. 누락된 경우
  기본 circular를 적용하고 해당 contig를 manifest에 기록합니다. `--unknown-topology linear`로 변경 가능합니다.

너무 작은 fragment에서는 UniOP가 strand 간 거리 분포를 추정할 수 없습니다.
이 경우 기본 모드는 오류를 내며 완성 유전체를 요구합니다.
명시적인 `--operon-mode singletons`는 모든 CDS를 독립 operon으로 처리하는 진단용 옵션입니다.
이 옵션 결과에 원래 benchmark 정확도를 적용하지 마세요.
13,000 genes / 9,000 operons를 넘으면 조용히 자르지 않고 실패합니다.

## 바로 실행할 수 있는 합성 예제

```bash
python prepare_inputs.py --genbank examples/example.gbk \
  --medium ISP2 --out outputs/demo --device cpu
```

`examples/example.gbk`와 `example.fna`는 검증용 합성 유전체입니다.
`python scripts/create_example_genome.py`로 재생성할 수 있습니다.
생물학적 성능 평가 데이터가 아니며 입력 생성 경로를 확인하는 데 사용합니다.
