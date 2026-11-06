# GitHub에 직접 업로드하기

기본 구성은 **소스는 GitHub, 학습 모델·kNN 검색 데이터는 Hugging Face**입니다.
모델 업로드·다운로드 명령은 [HUGGINGFACE.md](HUGGINGFACE.md)에 있습니다.
현재 코드는 학습 없이 저장 모델을 로드하도록 구성되어 있습니다.

## 1. 빈 GitHub 저장소 만들기

GitHub에서 **New repository**를 선택하고 이름과 공개 범위를 지정합니다.
기존 로컬 파일을 올릴 예정이므로 README, .gitignore, license 자동 생성은 선택하지 않습니다.
생성 후 HTTPS 또는 SSH 저장소 주소를 복사합니다.

아래 명령은 로컬 `Final_release` 폴더에서 직접 실행합니다. `OWNER`와 `REPOSITORY`는 실제 값으로 바꾸세요.
현재 이 폴더는 Git 저장소로 초기화되지 않은 상태입니다.

```bash
git init -b main
git add .
git status --short
git diff --cached --stat
git commit -m "Initial release: pretrained actinomycete expression prediction"
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

작성자 설정이 없어 commit이 실패할 경우 본인의 이름과 이메일을 지정한 뒤 다시 commit합니다.

```bash
git config user.name "YOUR NAME"
git config user.email "YOUR EMAIL"
```

HTTPS 인증이 필요하면 GitHub에서 사용하는 인증 수단을 사용합니다. GitHub 계정 비밀번호 대신
personal access token 또는 credential manager를 사용하며, 토큰을 소스나 remote URL에 넣지 않습니다.
SSH 인증이 이미 설정되어 있으면 `git@github.com:OWNER/REPOSITORY.git` 주소를 사용해도 됩니다.

`.gitignore`는 `config.local.json`, `artifacts/`, `data/`, `outputs/`, `dist/`,
`downloads/`, 가중치 및 캐시를 제외합니다. 위 `git status`에는 소스·문서·작은 예제와 평가 JSON만 보여야 합니다.
소스 ZIP을 단일 파일로 commit하는 대신, ZIP 안의 실제 소스 파일이나 현재 폴더를 올립니다.

## 2. 대안: 모델을 GitHub Release로 배포할 때만

Hugging Face 방식을 사용하면 이 절의 분할 명령은 실행하지 않습니다.
기존 `dist/Final_release-model.tar`를 Release에 배포할 경우에만 아래를 따릅니다.
소스에 이 파일을 commit하지 않습니다. GitHub는 일반 Git 파일에 100 MiB 제한을 두며,
Release 첨부 파일 하나도 2 GiB보다 작아야 합니다.
[일반 파일 제한](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github),
[Release 파일 제한](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)

기존 번들을 1,800 MiB씩 분할합니다. 이 명령은 GPU를 사용하지 않고 파일을 나누며,
원래 tar 외에 번들 크기만큼의 추가 디스크 공간이 필요합니다.

```bash
mkdir dist/model_release
split -b 1800M -d -a 2 dist/Final_release-model.tar \
  dist/model_release/Final_release-model.tar.part-
(cd dist/model_release && sha256sum Final_release-model.tar.part-* > SHA256SUMS)
```

분할 폴더가 이미 있으면 새 경로를 정하거나 기존 파일을 확인한 뒤 진행하세요.
현재 번들은 6개 part 파일로 나뉩니다.

GitHub 저장소의 **Releases → Draft a new release**에서 `v1.0.0` 같은 태그를 만들고,
`dist/model_release/` 안의 part 파일 전부와 `SHA256SUMS`를 첨부한 뒤 게시합니다.
README에도 해당 Release의 실제 링크를 추가하면 사용자가 모델을 찾을 수 있습니다.
모델을 별도 호스팅에 올릴 경우에는 tar 하나와 체크섬을 제공하고 README에 실제 주소를 연결해도 됩니다.

모델 번들에는 head 가중치 외에 training reference의 검색용 특징과 배지별 발현량 테이블,
유전자 ID 매핑 및 평가 결과가 들어 있습니다. 이를 빼면 현재 retrieval 모델을 그대로 사용할 수 없습니다.

## 3. 대안: GitHub Release 분할 파일을 받은 경우

소스를 clone한 폴더의 `downloads/`에 part 파일 전부와 `SHA256SUMS`를 저장합니다.
저장소 루트에서 실행합니다.

```bash
(cd downloads && sha256sum -c SHA256SUMS)
cat downloads/Final_release-model.tar.part-* | tar -xf -
sha256sum -c artifacts/SHA256SUMS
python -c "from model_artifacts import check_model; print(check_model('artifacts/expression'))"
```

첫 체크섬 검사가 실패하면 복원을 진행하지 않고 파일을 다시 받습니다.
복원 후에는 README의 입력 생성·예측 명령을 사용합니다. `train.py`는 실행하지 않습니다.

현재 저장소에는 새 라이선스를 임의로 지정하지 않았습니다. 공개 시 적용할 라이선스는
기존 코드와 데이터의 권리 조건에 맞춰 저장소 소유자가 정합니다.
