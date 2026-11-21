# GitHub Releases로 모델 배포하기

소스와 문서는 기존 [MYClN 저장소](https://github.com/kaist-sbml/MYClN)에서 관리하고,
학습 모델과 kNN 검색 데이터는 같은 저장소의 **Releases 첨부 파일**로 배포합니다.
아래 명령은 저장소 루트에서 직접 실행합니다.

## 1. 모델 파일 준비

```bash
python scripts/package_model.py &&
mkdir dist/model_release &&
split -b 1900M -d -a 2 dist/Final_release-model.tar \
  dist/model_release/Final_release-model.tar.part- &&
(cd dist/model_release && sha256sum Final_release-model.tar.part-* > SHA256SUMS)
```

이 명령은 로컬에서 모델을 묶고 분할하며 업로드하지 않습니다.
`dist/model_release/`가 이미 있으면 새 폴더명을 정하고 위 명령의 경로를 함께 바꾸세요.
기존 파일과 새 조각을 섞지 않도록 새 디렉터리를 사용합니다.

각 조각은 최대 1,900 MiB입니다. GitHub Releases의 첨부 파일은 각각 2 GiB 미만이어야 합니다.
[GitHub 공식 제한](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)

모델 전체가 약 9.8 GB이면 조각은 약 5개입니다. tar와 조각을 새로 만들 때 모델 크기의
약 두 배에 해당하는 추가 디스크 공간이 필요합니다.
번들에는 추론에 필요한 가중치, 검색 인덱스, 발현량 참조 테이블과 정규화 통계가 함께 들어갑니다.

## 2. 릴리스 게시

1. 배포할 소스와 문서를 GitHub에 반영합니다.
2. 저장소의 **Releases → Draft a new release**를 엽니다.
3. 해당 소스 커밋을 대상으로 `v1.0.0` 같은 새 버전 태그를 지정합니다.
4. `dist/model_release/`의 **모든 part 파일과 `SHA256SUMS`**를 첨부합니다.
5. 업로드 완료 후 릴리스를 게시합니다.

태그는 예시이며 실제 게시할 버전으로 정합니다. 모델 파일은 Release 첨부로 배포하므로
`artifacts/`와 `dist/`는 계속 `.gitignore`에 둡니다.

## 3. 사용자 다운로드와 복원

[Releases](https://github.com/kaist-sbml/MYClN/releases)에서 사용할 버전의
part 파일 전부와 `SHA256SUMS`를 받습니다. 같은 버전의 소스를 준비하고,
저장소 안의 새 `downloads/` 폴더에 다운로드한 파일을 놓습니다.
저장소 루트에서 실행합니다.

```bash
(cd downloads && sha256sum -c SHA256SUMS) &&
cat downloads/Final_release-model.tar.part-* | tar -xf - &&
sha256sum -c artifacts/SHA256SUMS
```

첫 체크섬 검사가 실패하면 복원은 진행되지 않습니다. 누락되거나 손상된 조각을 다시 받으세요.
복원 후 저장 모델을 확인할 수 있습니다.

```bash
python -c "from model_artifacts import check_model; print(check_model('artifacts/expression'))"
```

이후 [README](../README.md)의 예측 명령을 실행합니다. 재학습은 필요 없습니다.
