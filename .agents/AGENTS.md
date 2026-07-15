# Deep Vision Insights — AI Agent Guidelines

Computer Vision / Domain Adaptation 분야 논문 리뷰를 게시하는 학술 블로그.

## Tech Stack

- **SSG**: Hugo 0.147.0 (GitHub Actions), 로컬 빌드 테스트는 hugo 0.124.1
- **Theme**: PaperMod (git submodule: `themes/PaperMod`)
- **Math**: KaTeX 0.16.8 (`layouts/partials/extend_head.html`)
- **Hosting**: GitHub Pages (main 브랜치 push → Actions 자동 빌드/배포)
- **Languages**: en / ko 이중 언어 (`defaultContentLanguage = "en"`)

## Directory Structure

```
content/posts/           # 포스트 파일 (*.ko.md, *.en.md)
static/images/<slug>/    # 포스트별 이미지 디렉토리
layouts/partials/        # 커스텀 partial 오버라이드 (★ _partials 아님!)
themes/PaperMod/         # 테마 (submodule — 직접 수정 금지)
converted_mds/           # 논문 PDF를 변환한 원본 마크다운 소스
original_pdfs/           # 논문 PDF 원본
hugo.toml                # Hugo 설정
```

## 포스트 작성 워크플로우

### Step 1: PDF → 마크다운 변환

사용자가 `original_pdfs/` 디렉토리에 논문 PDF를 넣으면, 프로젝트 루트의 가상환경(`.venv`)에 `uv`로 설치된 **marker** 패키지를 사용하여 자동으로 마크다운으로 변환한다.

```bash
# marker_single로 단일 PDF 변환
.venv/bin/marker_single <PDF 경로> --output_dir converted_mds/
```

변환된 마크다운은 `converted_mds/` 디렉토리에 저장되며, 이를 원본 소스로 삼아 포스트를 작성한다.

### Step 2: 아티팩트(Artifact)로 초안 작성 및 피드백

> **중요: `content/posts/`에 `.md` 파일을 사용자 승인 없이 직접 생성하지 말 것.**

포스트 초안은 반드시 **Antigravity 아티팩트**(편집기 우측 패널에 렌더링되는 마크다운 문서)로 먼저 작성한다. 아티팩트에는 사용자가 인라인 코멘트를 달 수 있으므로, 이를 통해 내용·구조·문체에 대한 피드백을 주고받는다.

- 아티팩트 생성 시 `RequestFeedback: true`를 설정하여 사용자 승인 버튼을 표시한다.
- 아티팩트 내 수식이 깨져 보이는 것은 뷰어의 한계이므로 무시한다 (아래 LaTeX 섹션 참조).

### Step 3: 최종 승인 후 배포

사용자가 아티팩트를 확인하고 **"올려" / "배포해" / "커밋해"** 등의 최종 승인을 명시적으로 내린 후에만:

1. `content/posts/<slug>-review.ko.md`와 `<slug>-review.en.md`를 생성한다.
2. 이미지를 `static/images/<slug>/`에 배치한다.
3. `hugo` 로컬 빌드로 에러 여부를 확인한다.
4. `git add` → `git commit` → `git push`로 배포한다.

---

## Post Convention

### Frontmatter (필수 필드)

```yaml
---
title: "[학회 연도] 약칭: 한국어 제목 (ko) / 영문 제목 (en)"
date: YYYY-MM-DDTHH:MM:SS+09:00   # 반드시 작성 시점의 현재 날짜/시각
draft: false
math: true                         # 수식이 포함된 포스트는 반드시 true
tags: ["Paper Review", "키워드1", "키워드2", "학회 연도"]
categories: ["Paper Review"]
summary: "1~2문장 요약"
cover:
  image: "/images/<slug>/대표이미지.jpeg"
  alt: "대표 이미지 설명"
---
```

### 파일 이름 규칙

- 한국어: `<slug>-review.ko.md`
- 영어: `<slug>-review.en.md`
- 반드시 ko/en 쌍으로 생성할 것

### 본문 구조

1. 한 줄 요약
2. 연구 배경 및 동기 (2.1 문제 정의, 2.2 기존 방법의 한계, **2.3 핵심 기여**)
3. 제안 방법 프레임워크 (수식·그림 포함)
4. 실험 결과
5. 핵심 기여 정리 (2.3과 중복이 아닌, 상세 서술 버전)

> **주의**: 이 구조에 명시된 5가지 단락을 벗어난 6번째 단락을 만드는 것을 극히 삼갈 것. 골조를 불가피하게 못 맞춘다면 우선 재량껏 생성한 뒤, 본문 구조를 예외 생성했음을 불가피한 이유와 함께 사용자에게 알릴 것.
> 2단락의 핵심 기여와 5단락의 핵심 기여 정리는 둘 다 반드시 포함해야 한다. 2.3은 요약형 리스트, 5단락은 각 기여에 대한 상세 설명.

### 이미지 참조

- 경로: `/images/<slug>/파일명.jpeg` (절대 경로, `static/` 접두사 없음)
- 이미지 파일은 `static/images/<slug>/`에 저장
- 캡션: `*Figure N: 설명*` 형식 (이미지 바로 아래에 기재)

### 문체 규칙

- 번역투를 쓰지 않는다. 자연스러운 한국어로 정리한다.
- 원문을 직역하지 않는다. 핵심 내용을 간결하게 재구성한다.
- 총평이나 개인적 논평은 넣지 않는다. 논문 내용 전달에 충실한다.

---

## ⚠️ LaTeX 수식 렌더링 — 핵심 주의사항

> 이 프로젝트에서 수식을 다룰 때는 **두 가지 서로 다른 렌더링 환경**을 반드시 구분해야 한다.
> 과거 이 구분을 무시하여 수식이 깨지는 사고가 반복적으로 발생했다.
> **테스트 의무**: 새로 작업에 투입될 경우, 하기 기술된 방법대로 라텍스 테스트 아티팩트와 테스트 포스트(github mcp를 이용해 알아서 푸쉬할 것)를 생성한 뒤 무결성을 스스로 검토할 것.

### 환경 1: Antigravity 아티팩트 뷰어 (편집 중 미리보기)

- 마크다운 파서가 `_`(언더스코어)를 **이탤릭 구문**으로 해석한다.
- 따라서 인라인 수식 `$z_i^m$`이 `z`*`i`*`^m`처럼 깨진다.
- `**`(볼드) 마커 역시 파서가 먹어버려 `*` 기호가 원문에 노출될 수 있다.
- `$$` 블록 수식은 정상 렌더링된다.
- `\( \)` 인라인 구문은 일부 환경에서 동작하지 않는다.

### 환경 2: Hugo 블로그 (KaTeX 렌더링, 실제 배포 환경)

- `hugo.toml`에 passthrough 설정으로 `$`, `$$`, `\(`, `\)`, `\[`, `\]` 모두 수식 구분자로 등록되어 있다.
- KaTeX auto-render가 페이지 로드 시 수식을 렌더링한다.
- Goldmark(Hugo의 마크다운 파서)가 passthrough를 활성화했으므로 `_`를 이탤릭으로 해석하지 않는다.
- **따라서 `$z_i^m$` 같은 표준 LaTeX가 그대로 정상 동작한다.**

### 작성 원칙 (이 규칙을 어기면 수식이 깨진다)

1. **포스트 파일(`content/posts/*.md`)에는 표준 LaTeX를 그대로 쓴다.**
   - 수식은 라텍스 문법으로 작성함을 원칙으로 한다. 유니코드로 작성하여 포스트의 완성도를 떨어뜨리는 것을 엄금한다.
   - 블록: `$$ ... $$`
   - `\_`로 이스케이프하지 않는다. Hugo passthrough가 처리한다.

2. **★핵심★ 아티팩트용 / 포스팅용 수식 문법 이원화 원칙**
   - 아티팩트 뷰어에서 인라인 수식의 `_` 기호를 이탤릭체로 오인하여 수식이 깨지는 문제를 원천 차단하기 위해, **아티팩트 작성 시에는 반드시 HTML 태그(`<sub>`, `<sup>`, `<b>`) 및 유니코드 기호 등을 사용합니다.**
   - **절대 금지**: 아티팩트 내에서 `$z_i^m$`, `$X_H$` (언더스코어 `_` 사용 시 아티팩트 깨짐 발생)
   - **필수 준수**: 아티팩트 내에서는 **z<sub>i</sub><sup>m</sup>**, **X<sub>H</sub>** 등 시각적으로 완벽하게 렌더링되는 HTML/유니코드를 사용합니다.
   - 이원화 관리를 통해 아티팩트 검토 시에는 수식 깨짐 없는 깨끗한 문서를 확인하고, 추후 블로그 포스트 `.md` 파일을 생성할 때만 별도의 파이썬 스크립트를 통해 표준 포스팅용 문법(LaTeX)으로 일괄 변환하여 무결성을 유지합니다.

3. **아티팩트 수식 깨짐 방지**
   - 위 2번 규칙(HTML/유니코드 기호)을 준수하여 아티팩트 내 수식 깨짐을 방지합니다.
   - 단, 아티팩트용으로 사용된 이스케이프(`\_`), HTML 태그(`<sub>`, `<sup>`), 유니코드 문자 등은 **Hugo 블로그에서 수식을 망가뜨리므로 최종 포스트 파일에는 절대 커밋하지 않습니다.**

4. **사용자에게 수식 미리보기를 보여줘야 할 때:**
   - 아티팩트의 수식이 깨져 보이는 이유를 설명한다.
   - "블로그에서는 정상 렌더링됩니다"라고 안내한다.
   - 필요하면 `hugo server`로 로컬 빌드 후 브라우저 캡처로 증명한다.

4. **절대 하지 말 것:**
   - `\_`를 써서 언더스코어를 이스케이프하는 것
   - **최종 포스트 파일**에 `<sub>`, `<sup>` 등 HTML 태그로 수식을 대체하여 남겨두는 것
   - **최종 포스트 파일**에 유니코드 특수문자(∑, Γ, ℒ 등)로 LaTeX를 대체하여 남겨두는 것
   - `\( \)` 인라인 구문을 아티팩트에서만 쓰고 커밋할 때 `$ $`로 바꾸는 이중 관리

---

## Deployment

### 커밋 & 푸시 순서

```bash
# 1. 이미지 먼저 추가
git add static/images/<slug>/

# 2. 포스트 파일 추가
git add content/posts/<slug>-review.ko.md content/posts/<slug>-review.en.md

# 3. 커밋 & 푸시
git commit -m "Add <논문명> paper review post"
git push
```

### 빌드 검증

```bash
# 로컬 빌드 테스트 (에러 없이 완료되어야 함)
hugo

# 에러 발생 시 로그 확인 후 수정 → 재커밋
```

### 주의사항

- `themes/PaperMod`은 git submodule이다. 내부를 직접 수정하면 커밋이 꼬인다.
- `layouts/partials/`는 Hugo가 인식하는 정규 경로이다. **절대 `_partials`로 바꾸지 말 것.** (`_partials`로 바꾸면 `partial "head.html" not found` 에러로 전체 사이트 빌드가 실패한다.)
- GitHub Actions의 Hugo 버전(0.147.0)과 로컬 Hugo 버전(0.124.1)이 다르다. 로컬에서 Warning이 나와도 Actions에서 정상 빌드되면 문제없다.

## Do

- 포스트 초안은 반드시 아티팩트(Artifact)로 먼저 작성하며, **모든 포스트 글을 우선 국문으로 작성**한다. 이후 사용자 피드백을 거쳐 최종 승인을 받아 `.md` 파일을 생성한다.
- `math: true`를 frontmatter에 반드시 포함한다 (수식이 있는 포스트).
- `date` 필드에 작성 시점의 현재 날짜를 넣는다.
- 배포 전 `hugo` 로컬 빌드로 에러가 없는지 확인한다.
- 커밋 전 `git status`로 의도하지 않은 파일이 포함되지 않았는지 확인한다.

## Don't

- 사용자의 명시적 최종 승인 없이 `content/posts/`에 `.md` 파일을 생성하지 않는다.
- **ko/en 쌍을 매 시행 불필요하게 생성하여 토큰을 낭비하지 말 것.** 국문 최종 포스트 글이 완성된 이후에 그것을 en으로 번역하여 푸쉬한다.
- **수식은 라텍스 문법으로 작성함을 원칙으로 한다.** 단, 아티팩트 뷰어용으로는 HTML/유니코드를 활용한 이원화를 진행하며, 최종 포스트에는 라텍스 문법으로 원상 복구한다. 유니코드로 포스트의 완성도를 떨어뜨리는 것을 엄금한다.
- `themes/PaperMod/` 내부 파일을 직접 수정하지 않는다.
- `layouts/partials`를 `layouts/_partials`로 바꾸지 않는다.
- 사용자가 명시적으로 요청하지 않은 기존 포스트를 수정하지 않는다.
- 아티팩트 미리보기에서 수식이 깨진다고 **최종 포스팅용** LaTeX 문법을 변형하지 않는다.
- `git config user.email`을 임의의 값으로 설정하지 않는다.
