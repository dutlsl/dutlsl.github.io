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

## ⚠️ LaTeX Math Rendering — Critical Warnings

> Two rendering environments must be strictly distinguished when handling math equations. Ignoring this distinction previously led to repeated rendering issues.
> **Testing Obligation**: Before starting work on this repo, create a LaTeX test artifact and a test post (push it automatically using GitHub MCP) to verify rendering integrity.

### Environment 1: Antigravity Artifact Viewer (Preview in Editor)

- The markdown parser interprets `_` (underscores) as italic markdown markers.
- Thus, inline math like `$z_i^m$` gets broken into `z`*`i`*`^m`.
- Bold markers (`**`) may also be swallowed, exposing raw `*` symbols.
- `$$` block equations render normally.
- `\( \)` inline syntax is not supported in some environments.

### Environment 2: Hugo Blog (KaTeX Rendering, Live Production)

- The `hugo.toml` configuration registers `$`, `$$`, `\(`, `\)`, `\[`, and `\]` as math delimiters via passthrough.
- KaTeX auto-render renders math dynamically on page load.
- Goldmark (Hugo's MD parser) has passthrough enabled, so it does not interpret `_` as italics inside delimiters.
- **Therefore, standard LaTeX like `$z_i^m$` works perfectly out of the box.**

### Writing Principles (To Prevent Math Breakage)

1. **Use standard LaTeX inside post files (`content/posts/*.md`).**
   - Write math equations in standard LaTeX. Do not fall back to unicode symbols for post quality.
   - Block math: `$$ ... $$`
   - Do not escape underscores with `\_`. The Hugo passthrough will handle it correctly.

2. **Isomorphic Math Rule for Artifacts vs. Posts (CRITICAL)**
   - To prevent underscores from breaking inline math in the artifact viewer, **always use HTML tags (`<sub>`, `<sup>`, `<b>`) and unicode symbols for math in artifacts.**
   - **ABSOLUTELY PROHIBITED**: Do not use standard LaTeX syntax with underscores (e.g., `$z_i^m$`, `$X_H$`) in artifacts, as they will break.
   - **MANDATORY**: In artifacts, write math using HTML/Unicode (e.g., **z<sub>i</sub><sup>m</sup>**, **X<sub>H</sub>**).
   - Convert these HTML representations back to standard LaTeX when generating the final `.md` post files.

3. **Prevent Artifact Math Breakage**
   - Follow Rule 2 to keep artifacts clean. However, **never commit HTML tags or unicode symbols in place of LaTeX inside the final `.md` files**, as they will break KaTeX rendering on the live blog.

4. **Showing Math Previews to Users**
   - Explain the viewer's math rendering limitations.
   - Assure them it renders fine on the blog, and verify by running `hugo server` locally and capturing a screenshot if needed.

5. **ABSOLUTELY PROHIBITED**:
   - Do not escape underscores with `\_`.
   - Do not leave HTML tags (`<sub>`, `<sup>`) in the final post files.
   - Do not leave unicode representations of LaTeX in the final post files.
   - Do not mix `\( \)` in artifacts only to rewrite them to `$ $` during commit.

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
