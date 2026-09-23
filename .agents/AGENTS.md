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

1. One-Sentence Summary: Condense the core value and contribution of the paper into a single impactful sentence.
2. Research Background and Motivation:
   - 2.1 Problem Definition (Opening with a relatable everyday analogy to build empathy → formal technical problem definition)
   - 2.2 Limitations of Existing Methods (Analysis of critical bottlenecks and failure causes in prior methods)
   - 2.3 Main Contributions (Present the three core contributions in clear bullet points)
3. Proposed Framework: Pipeline overview tied to the penetrating analogy, followed by module-by-module mechanisms (analogy-formula-term decomposition sandwich)
4. Experimental Results: Quantitative and qualitative evaluations, key comparative benchmark tables, ablation study analyses
5. Conclusion and Key Takeaways: Rather than simply repeating 2.3, expand narratively into the meta-insights and paradigm shift delivered to the field

> **WARNING**: Strictly avoid adding a 6th section outside this 5-section structure. If an exception is absolutely necessary, construct it at your discretion first, then inform the user with the rationale for the exception.
> Both Section 2.3 (summary list) and Section 5 (detailed narrative explanations) must be included.

### Image Reference

- Path: `/images/<slug>/filename.jpeg` (Absolute path, no `static/` prefix)
- Store image files in `static/images/<slug>/`
- Caption: Use the `*Figure N: Description*` format directly below the image.

### Writing Strategy & Narrative Style

- **Penetrating Analogy**: Establish an intuitive everyday analogy that consistently threads through the entire research from introduction to conclusion. Readers without domain expertise should grasp the core intuition immediately.
  - **Maintain Domain Diversity**: Avoid reusing analogy domains (e.g., cooking/kitchen) across multiple posts. Discover fresh everyday domains (e.g., architecture, transit/traffic, sports, manufacturing) tailored to each paper's unique mechanism.
- **Reader-Engaging Opening (Section 2.1)**: Rather than front-loading formulas or technical jargon, open with 3–4 sentences describing a relatable everyday scenario or dilemma to naturally immerse the reader in why the research is necessary.
- **Analogy-Formula-Decomposition Sandwich (Section 3)**:
  - Explain the intuitive analogy and physical intuition before introducing any equation.
  - Present the formula.
  - Preemptively and completely decompose each term in the equation—explaining its physical meaning, rationale for introduction, basis for signs (+, -), and necessity of regularization or constraints—before the reader can form doubts. Never gloss over, skip, or brush aside the most challenging mathematical parts.
- **Narrative Conclusion (Section 5)**: Never simply repeat the bullet points from Section 2.3. Expand into a completed narrative discussing the meta-insights and paradigm-shifting value the proposed methodology delivers to academia and industry.
- **Tone & Readability Principles**:
  - Unify the writing in polite Korean honorifics (ending with -합니다/-입니다), strictly eliminating unnatural translationese.
  - Keep sentences short and clear (1–2 lines per sentence), avoiding convoluted compound sentences.
  - Articulate details in full, standalone sentences without cramming information into parentheses.
  - Strictly prohibit bold formatting (`**`, `<b>`) in Korean text to prevent editor rendering breakage.

---

## ⚠️ Artifact Image & Math Insertion Rules (Mandatory Global Rules)

To eliminate rendering discrepancies between the Antigravity artifact viewer (editor right panel) and the production Hugo blog (KaTeX), strictly follow these rules:

### 1. Artifact Image Insertion Rules — ★MANDATORY★

To ensure 100% compatibility with the Antigravity artifact viewer and Hugo blog, the following single standard must be strictly enforced. All alternative formats are strictly prohibited.

1. **File Copy and Permission Setup (Mandatory Prerequisite)**:
   - All extracted image files (`_page_X_...jpeg`) must be **immediately** copied to two locations simultaneously with full permissions (`chmod -R 777`):
     1. Directly under the current conversation's **artifact root directory** (`/home/iulab1/.gemini/antigravity-ide/brain/<conversation-id>/`)
     2. The blog static asset directory (`/home/iulab1/dutlsl.github.io/static/images/<slug>/`)
   - Leaving images solely in an `images/` subfolder without copying them directly to the artifact root directory is strictly prohibited.

2. **Frontmatter Cover Image**:
   - The YAML frontmatter `cover.image` **must always use the blog standard web path**:
     ```yaml
     cover:
       image: "/images/<slug>/filename.jpeg"
       alt: "Representative image description"
     ```
   - The first line and first column of the file must start with `---`. Never place any title or content before `---`.

3. **Mandatory Top Overview Figure in Body**:
   - Because `cover.image` in the frontmatter does not render at the top of the artifact preview, **the representative overview diagram (Figure 1) must explicitly be placed as the very first figure in the body, right below the paper metadata block (`> Reference Paper`)**:
     ```markdown
     > Reference Paper
     > - ...

     ![Figure 1: Overview](/home/iulab1/.gemini/antigravity-ide/brain/<conversation-id>/filename.jpeg)
     *Figure 1: Detailed caption description*

     ---

     ## 1. One-Sentence Summary
     ```

4. **Image Link Syntax in Artifact Body**:
   - **Always use the absolute path to the artifact root directory**:
     ```markdown
     ![Figure N: Concise Title](/home/iulab1/.gemini/antigravity-ide/brain/<conversation-id>/filename.jpeg)
     *Figure N: Detailed caption description*
     ```
   - **Prohibitions**:
     - Do NOT use subfolder paths (e.g. `/home/.../images/...`), which trigger Webview CSP blocking.
     - Do NOT use web relative paths (e.g. `/images/<slug>/...`) in artifact drafts (interpreted as OS root `/images/` by the editor, causing broken image icons).
     - Do NOT use `file:///` scheme prefixes arbitrarily. Standardize strictly on `/home/iulab1/...` absolute paths.
   - Keep the alt text inside `![...]` concise (e.g. `![Figure N: Title]`), and place any detailed descriptions on the line immediately below using italic text (`*Figure N: Description*`).

5. **Conversion to Production Blog Markdown**:
   - Only upon final user approval, when generating the production post files (`content/posts/<slug>-review.ko.md` and `.en.md`), convert all body image paths to Hugo web paths (`/images/<slug>/filename.jpeg`).

---

### 2. Artifact vs. Blog Math Formula Rules

| Scope | Artifact Viewer (Editor Right Panel) | Production Hugo Blog (`content/posts/*.md`) |
|---|---|---|
| **Syntax** | **100% HTML tags + Unicode** | **100% Standard LaTeX (`$ ... $`, `$$ ... $$`)** |
| **Examples** | `z<sub>i</sub><sup>m</sup>`, `ℝ<sup>H×W</sup>`, `θ → 0` | `$z_i^m$`, `$\mathbb{R}^{H \times W}$`, `$\theta \to 0$` |
| **Prohibitions** | Never use `$ ... $`, `$$ ... $$`, or `\_` (breaks Markdown parser) | Never leave `<sub>`, `<sup>`, or Unicode approximations (breaks KaTeX) |

1. **Inside Artifacts**:
   - The editor artifact viewer misinterprets LaTeX `_` (underscore) as markdown italic (`_text_`), corrupting entire equations and surrounding paragraphs.
   - Therefore, inside artifacts, **strictly avoid `$` and `$$` delimiters**, and express indices/exponents with `<sub>` and `<sup>` tags and operators/Greek letters with Unicode symbols (α, β, θ, σ, λ, ×, ∈, ℝ, →, etc.).
2. **Inside Production Blog Posts**:
   - When generating the final `content/posts/<slug>-review.ko.md` and `.en.md` after user approval, convert all HTML math representations **back to clean standard LaTeX (`$ ... $`, `$$ ... $$`)**.
   - Hugo KaTeX has Goldmark passthrough configured, guaranteeing flawless LaTeX rendering on the live blog.

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

- **Benchmark Golden References First**: Before drafting, read and benchmark the narrative flow, analogy sandwich structure, and decomposition style of the gold standard posts:
  - `content/posts/fevos-review.ko.md` (Best-in-class analogy-formula-decomposition sandwich, pre-emptive term breakdown, narrative conclusion)
  - `content/posts/segfs-review.ko.md` (Seamless penetrating analogy threading entire paper, relatable opening)
- Draft posts as artifacts first, **writing the initial draft in Korean**. Generate the `.md` files only after receiving final user approval.
- Always include `math: true` in the frontmatter for posts containing math.
- Use the current date/time for the `date` field.
- Verify the build locally with `hugo` before deploying.
- Run `git status` before committing to avoid staging unintended files.

## Don't

### Category A: Style & Expression Prohibitions

- **No Bold Text in Korean Posts and Artifacts**: Using `**` or `<b>` tags in Korean posts and artifacts is strictly prohibited to prevent rendering issues and raw asterisks from being displayed.
- **Strictly prohibit the use of parentheses and cramming information**: Cramming supplementary explanations, parameter values, or English translations inside parentheses is strictly prohibited. State all details and numerical values in complete, natural, and flowing sentences without parentheses. Avoid parentheses entirely in body text unless technically unavoidable (e.g. math formulas or markdown links).
- **Do not force Korean translations or translations alongside common English terminology**: For terms widely accepted and used in the industry (e.g., *latency*, *strictly causal constraint*, *visual saliency*, *gaze fixation*, *gaze diffusion*, *target ambiguity*, *isomorphic*, *motion blur*, etc.), use the English term directly instead of forced Korean translations or pairing them together (e.g., pairing Korean and English like "지연 시간(Latency)").
- **Avoid excessively long, complex, or compound sentence structures**: Keep sentences short, concise, and clear (ideally within 1–2 lines). Break down long sentences with multiple conjunctions or commas to improve readability.
- **Do not use cliché, robotic, or AI-generated tones and bulleted noun headers**: Avoid formulaic templates, robotic intro/connecting sentences (e.g., "This constraint poses two key questions..."), and AI-like nominalized headers (e.g., rhetorical question headers or dry noun phrases like "~의 효용성: ~하는가?" or "~의 비단조성"). Write in a natural, cohesive, and narrative style suitable for editorial/essay blog posts.
- **Do not bypass or gloss over difficult math terms**: Never skip or brush over the explanation of complex terms, signs, or regularization factors in formulas; break them down completely with intuitive rationale.
- **Avoid repeating the same analogy domain across posts**: Do not reuse analogy domains (e.g., cooking/kitchen) from previous posts; discover fresh everyday domains fitting each paper.

### Category B: Structure & Visualization Prohibitions

- **Strictly Follow the Isomorphic Math Rule**: Never use `$ ... $` or `$$ ... $$` delimiters for math inside artifacts. Use 100% HTML tags (`<sub>`, `<sup>`) and text/unicode equivalents.
- **Always write math equations in standard LaTeX inside the final post files**: Do not corrupt the post quality with unicode approximations in production markdown. Do not compromise the LaTeX syntax in the final post files to fix artifact preview breakage.
- **Artifact-Specific Image Path Rule (Strictly Enforced)**:
  - Extracted images must **immediately be copied directly under the artifact root directory** (`<appDataDir>/brain/<conversation-id>/`) and the static asset directory (`static/images/<slug>/`) with full permissions (`chmod -R 777`).
  - In YAML frontmatter, `cover.image` must always use the blog web path (`/images/<slug>/filename.jpeg`).
  - The representative overview figure (Figure 1) must explicitly be placed as the first figure directly below the paper reference metadata block.
  - All image references in the artifact body **must strictly use the absolute path to the brain root directory** (`![Title](/home/iulab1/.gemini/antigravity-ide/brain/<conv_id>/_page_X_...jpeg)`).
  - Using subfolder paths (e.g. `/home/.../images/...`) or web relative paths (`/images/...`) in the artifact body is strictly prohibited.
  - Only upon final user deployment approval, convert all body paths to `/images/<slug>/...` when writing the final `.md` files.
- **Ensure consistency between model components in figures and textual descriptions**: When describing model architectures, strictly match the names of components used in the text (e.g., `Frozen DINOv3`, `Shared Spatio-Temporal Decoder`, `GLF & Conv Head`) with those labeled in the corresponding figures to prevent confusion. Do not invent arbitrary names.
- **Prohibit Mermaid Diagrams**: Inserting Mermaid diagram code blocks or flowcharts into posts and artifacts is strictly prohibited. Flowcharts look mechanical and clash with the blog's aesthetic. Explain all pipelines and architectures strictly using the paper's official figures, structured markdown tables, and clear narrative prose.
- **No 6th section**: Strictly avoid adding a 6th section outside the 5-section body structure.

### Category C: Process & System Prohibitions

- Do not create `.md` files in `content/posts/` without explicit user approval.
- **Do not waste tokens by translating to English prematurely.** Only translate the post to English after the Korean draft has been finalized and approved.
- Do not modify files inside `themes/PaperMod/`.
- Do not rename `layouts/partials` to `layouts/_partials`.
- Do not modify existing posts unless explicitly requested.
- Do not configure `git config user.email` to an arbitrary value.
- **Review Strategy for Subsequent Draft Revisions**: Once a draft is submitted, do not restructure or rewrite the entire post based on user comments. Keep the original structure intact, and naturally integrate error fixes, answers to user queries, or term definitions into the existing text flow.


