# Deep Vision Insights — AI Agent Guidelines

An academic blog publishing paper reviews in the fields of Computer Vision and Domain Adaptation.

## Tech Stack

- **SSG**: Hugo 0.147.0 (GitHub Actions), local build tests with Hugo 0.124.1
- **Theme**: PaperMod (git submodule: `themes/PaperMod`)
- **Math**: KaTeX 0.16.8 (`layouts/partials/extend_head.html`)
- **Hosting**: GitHub Pages (push to main branch → auto build/deploy via Actions)
- **Languages**: Bilingual (en / ko) (defaultContentLanguage = "en")

## Directory Structure

```
content/posts/           # Post files (*.ko.md, *.en.md)
static/images/<slug>/    # Image directory per post
layouts/partials/        # Custom partial overrides (★ NOT _partials!)
themes/PaperMod/         # Theme (submodule — DO NOT modify directly)
converted_mds/           # Original markdown sources converted from paper PDFs
original_pdfs/           # Original paper PDFs
hugo.toml                # Hugo configuration
```

## Post Writing Workflow

### Step 1: PDF → Markdown Conversion

When a user places a paper PDF in the `original_pdfs/` directory, it must be automatically converted to markdown using the **marker** package installed in the project root's virtual environment (`.venv`) via `uv`.

```bash
# Convert a single PDF using marker_single
.venv/bin/marker_single <PDF path> --output_dir converted_mds/
```

The converted markdown is saved in the `converted_mds/` directory and serves as the raw source for writing the post.

### Step 2: Draft Creation & Feedback via Artifacts

> **IMPORTANT: Do NOT create any `.md` files directly in `content/posts/` without explicit user approval.**

Post drafts must always be written as an **Antigravity Artifact** (rendered in the right panel of the editor) first. Users can leave inline comments on this artifact to provide feedback on content, structure, and style.

- Set `RequestFeedback: true` in the ArtifactMetadata to display the user approval button.
- Ignore broken math formulas in the artifact preview; this is a viewer limitation (see the LaTeX section below).

### Step 3: Final Approval and Deployment

Only after the user reviews the artifact and gives explicit final approval (e.g., "deploy", "upload", or "commit"):

1. Create `content/posts/<slug>-review.ko.md` and `<slug>-review.en.md`.
2. Place the images in the `static/images/<slug>/` directory.
3. Verify that the build succeeds without errors using a local `hugo` build.
4. Run `git add` → `git commit` → `git push` to deploy.

---

## Post Convention

### Frontmatter (Required Fields)

```yaml
---
title: "[Venue Year] Abbreviation: Korean Title (ko) / English Title (en)"
date: YYYY-MM-DDTHH:MM:SS+09:00   # Must be the current date/time at the moment of creation
draft: false
math: true                         # Must be true if the post contains equations
tags: ["Paper Review", "Keyword 1", "Keyword 2", "Venue Year"]
categories: ["Paper Review"]
summary: "1~2 sentence summary"
cover:
  image: "/images/<slug>/representative_image.jpeg"
  alt: "Representative image description"
---
```

### Naming Conventions

- Korean: `<slug>-review.ko.md`
- English: `<slug>-review.en.md`
- Always create them as a ko/en pair.

### Body Structure

1. One-Sentence Summary
2. Research Background and Motivation (2.1 Problem Definition, 2.2 Limitations of Existing Methods, **2.3 Main Contributions**)
3. Proposed Framework (Include equations/figures)
4. Experimental Results
5. Conclusion and Key Takeaways (A detailed, narrative version of the contributions, not a duplicate of 2.3)

> **WARNING**: Strictly avoid adding a 6th section outside this 5-section structure. If an exception is absolutely necessary, construct it at your discretion first, then inform the user with the rationale for the exception.
> Both Section 2.3 (summary list) and Section 5 (detailed narrative explanations) must be included.

### Image Reference

- Path: `/images/<slug>/filename.jpeg` (Absolute path, no `static/` prefix)
- Store image files in `static/images/<slug>/`
- Caption: Use the `*Figure N: Description*` format directly below the image.

### Writing Style (For Korean Posts)

- The tone must be unified in polite, natural Korean (e.g., ending with -합니다/-입니다) rather than neutral -한다 style.
- Do not use translated tones (번역투). Write in natural Korean.
- Do not translate the source text literally. Reconstruct key ideas concisely.
- Do not include general personal commentary or opinions. Stay faithful to the paper's actual content.

---

## ⚠️ Artifact Image & Math Insertion Rules (전역 필수 하네스 규정)

아티팩트 뷰어(에디터 우측 패널)와 Hugo 블로그(웹 프로덕션)의 렌더링 엔진 차이로 인한 오류를 방지하기 위해 아래 규칙을 100% 엄격히 준수합니다.

### 1. 아티팩트 그림/이미지 삽입 규정 (Image Rules)

1. **파일 복사 및 권한 설정 (필수 사전 작업)**:
   - PDF에서 추출된 이미지들은 **즉시** 현재 아티팩트 디렉토리 (`<appDataDir>/brain/<conversation-id>/`) 및 `images/` 하위 폴더로 복사하고, 읽기 권한(`chmod -R 777`)을 부여해야 합니다.
   - 동시에 블로그 정적 디렉토리(`static/images/<slug>/`)에도 미리 복사해 둡니다.
2. **아티팩트 내 마크다운 문법 (Artifact Markdown Syntax)**:
   - **반드시 아티팩트 디렉토리의 절대 경로 사용**:
     ```markdown
     ![Figure 1: Architecture Overview](/home/iulab1/.gemini/antigravity-ide/brain/<conversation-id>/filename.jpeg)
     *Figure 1: 상세 설명*
     ```
   - **대괄호 `![...]` 안의 캡션은 짧고 명확하게 작성**: 과도하게 긴 문장이나 특수문자를 대괄호 안에 넣으면 파서에 따라 렌더링이 깨질 수 있으므로, 간결한 제목만 넣고 상세 설명은 바로 아래 줄의 `*Figure N: ...*` 기울임꼴 텍스트로 작성합니다.
   - **프론트매터 cover image**: 아티팩트에서는 `cover.image`에도 아티팩트 절대 경로를 지정합니다.
   - **프론트매터 최상단 배치**: 아티팩트 마크다운 파일의 1행 1열은 반드시 `---`로 시작해야 합니다. `---` 앞에 `# 제목` 등을 절대 넣지 않습니다.
3. **최종 블로그 `.md` 파일 변환 시**:
   - 최종 승인 후 `content/posts/`에 생성할 때만 모든 이미지 경로를 `/images/<slug>/filename.jpeg`로 일괄 치환합니다.

---

### 2. 아티팩트 vs 블로그 수식 삽입 규정 (Math Formula Rules)

| 구분 | 아티팩트 뷰어 (에디터 우측 프리뷰) | 최종 Hugo 블로그 (`content/posts/*.md`) |
|---|---|---|
| **문법** | **100% HTML 태그 + 유니코드** | **100% 표준 LaTeX (`$ ... $`, `$$ ... $$`)** |
| **표기 예시** | `z<sub>i</sub><sup>m</sup>`, `ℝ<sup>H×W</sup>`, `θ → 0` | `$z_i^m$`, `$\mathbb{R}^{H \times W}$`, `$\theta \to 0$` |
| **금지 사항** | `$ ... $`, `$$ ... $$`, `\_` 절대 사용 금지 (마크다운 파서 깨짐) | `<sub>`, `<sup>`, 유니코드 수식 잔존 절대 금지 (KaTeX 깨짐) |

1. **아티팩트 작성 시**:
   - 에디터 아티팩트 뷰어는 LaTeX의 `_` 기호를 마크다운 이탤릭(`_text_`)으로 오인하여 수식과 텍스트 전체를 파괴합니다.
   - 따라서 아티팩트 내부에서는 **절대로 `$`나 `$$` 구분자를 쓰지 않고**, 첨자는 `<sub>`, `<sup>` 태그로, 그리스 문자나 연산자는 유니코드 기호(α, β, θ, σ, λ, ×, ∈, ℝ, → 등)로 완전 대체하여 작성합니다.
2. **최종 블로그 파일 생성 시**:
   - 사용자의 최종 승인 후 `content/posts/<slug>-review.ko.md` 및 `.en.md`를 생성할 때는, 아티팩트의 HTML 수식을 **완전한 표준 LaTeX 문법(`$ ... $`, `$$ ... $$`)으로 역변환(환원)**하여 저장합니다.
   - Hugo KaTeX는 Goldmark passthrough 설정이 되어 있으므로 표준 LaTeX가 오류 없이 완벽하게 렌더링됩니다.

---

## Deployment

### Commit & Push Order

```bash
# 1. Add images first
git add static/images/<slug>/

# 2. Add post files
git add content/posts/<slug>-review.ko.md content/posts/<slug>-review.en.md

# 3. Commit and push
git commit -m "Add <Paper Name> paper review post"
git push
```

### Build Verification

```bash
# Run local build to verify no errors occur
hugo

# Check logs and resolve errors if they occur, then recommit
```

### Important Notices

- `themes/PaperMod` is a git submodule. Direct modifications inside this directory will break git commits.
- `layouts/partials/` is the standard Hugo layouts directory. **Do NOT rename it to `_partials`.** (Renaming it will cause `partial "head.html" not found` errors and break the site build).
- The Hugo version used in GitHub Actions (0.147.0) differs from the local version (0.124.1). Local warnings can be ignored as long as the Actions build succeeds.

## Do

- Draft posts as artifacts first, **writing the initial draft in Korean**. Generate the `.md` files only after receiving final user approval.
- Always include `math: true` in the frontmatter for posts containing math.
- Use the current date/time for the `date` field.
- Verify the build locally with `hugo` before deploying.
- Run `git status` before committing to avoid staging unintended files.

## Don't

- Do not create `.md` files in `content/posts/` without explicit user approval.
- **Do not waste tokens by translating to English prematurely.** Only translate the post to English after the Korean draft has been finalized and approved.
- **Always write math equations in standard LaTeX inside the final post files.** Do not corrupt the post quality with unicode approximations.
- Do not modify files inside `themes/PaperMod/`.
- Do not rename `layouts/partials` to `layouts/_partials`.
- Do not modify existing posts unless explicitly requested.
- Do not compromise the LaTeX syntax in the final post files to fix artifact preview breakage.
- Do not configure `git config user.email` to an arbitrary value.
- **No Bold Text in Korean Posts and Artifacts**: Using `**` or `<b>` tags in Korean posts and artifacts is strictly prohibited to prevent rendering issues and raw asterisks from being displayed.
- **Strictly Follow the Isomorphic Math Rule**: Never use `$ ... $` or `$$ ... $$` delimiters for math inside artifacts. Use 100% HTML tags (`<sub>`, `<sup>`) and text/unicode equivalents.
- **Artifact-Specific Image Path Rule**: To render images correctly in the artifact preview, copy the extracted images to the artifact directory (`<appDataDir>/brain/<conversation-id>/`) and reference them using absolute markdown paths (e.g., `![caption](/absolute/path/to/image.jpeg)`). Convert these paths back to `/images/<slug>/...` only when writing the final `.md` files.
- **Review Strategy for Subsequent Draft Revisions**:
  - Once a draft is submitted, do not restructure or rewrite the entire post based on user comments.
  - Keep the original structure intact, and naturally integrate error fixes, answers to user queries, or term definitions (e.g., prompt domain, spectral domain, shared mamba) into the existing text flow.
- **Do not force Korean translations or translations alongside common English terminology**: For terms widely accepted and used in the industry (e.g., *latency*, *strictly causal constraint*, *visual saliency*, *gaze fixation*, *gaze diffusion*, *target ambiguity*, *isomorphic*, *motion blur*, etc.), use the English term directly instead of forced Korean translations or pairing them together (e.g., 지연 시간(Latency)).
- **Avoid excessively long, complex, or compound sentence structures**: Keep sentences short, concise, and clear (ideally within 1–2 lines). Break down long sentences with multiple conjunctions or commas to improve readability.
- **Do not use cliché, robotic, or AI-generated tones and bulleted noun headers**: Avoid formulaic templates, robotic intro/connecting sentences (e.g., "This constraint poses two key questions..."), and AI-like nominalized headers (e.g., `~의 효용성: ~하는가?`, `~의 비단조성`). Write in a natural, cohesive, and narrative style suitable for editorial/essay blog posts.
- **Ensure consistency between model components in figures and textual descriptions**: When describing model architectures, strictly match the names of components used in the text (e.g., `Frozen DINOv3`, `Shared Spatio-Temporal Decoder`, `GLF & Conv Head`) with those labeled in the corresponding figures to prevent confusion. Do not invent arbitrary names.
- **괄호(소괄호) 사용 및 정보 욱여넣기 엄격 금지**: 괄호 안에 부가 설명, 보충 정보, 영문 번역, 파라미터 수치, 사족 등을 욱여넣는 작성을 전면 금지합니다 (예: `ViT 이전(before) 단계`, `경량(3M 파라미터) 모듈`, `지연 시간(latency)` 등 금지). 모든 내용과 수치는 괄호 없이 온전하고 자연스러운 문장 서술로 풀어써야 합니다. 수식 표기나 마크다운 링크 문법 등 기술적으로 불가피한 경우를 제외하고 본문 텍스트 내에서 소괄호 사용을 철저히 배제합니다.
- **머메이드(Mermaid) 다이어그램 작성 금지**: 포스트 및 아티팩트 내에 머메이드(mermaid) 코드 블록이나 플로우차트를 삽입하는 것을 엄격히 금지합니다. AI 특유의 기계적인 템플릿 느낌을 주며 블로그 렌더링에 이질감을 유발하므로, 파이프라인이나 아키텍처 흐름은 논문의 공식 그림과 명확하고 자연스러운 본문 서술, 구조화된 목록 또는 마크다운 표로만 설명해야 합니다.

