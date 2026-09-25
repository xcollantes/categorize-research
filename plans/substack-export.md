# Plan: Substack version of the project page (mixed approach)

## Context

`docs/index.html` (the GitHub Pages site) uses custom HTML/CSS: cards, a
facts grid, a price chart, coloured tags. Substack's editor doesn't accept
custom HTML or CSS, so pasting the page loses all of that. The goal is a
Substack post that keeps the visuals where they matter:

- **Native Substack text** for the prose, lists and prompt code blocks, so
  they stay selectable, searchable and copyable.
- **Images** for the visual components.
- **A link** back to the live page for the full interactive version.

Decisions from the user:
- Export now with the placeholder Results image. Rerun the script after real
  numbers go into `docs/index.html`.
- Script and output live in the repo and are committed.
- The Prompt section uses native code blocks, not images.

## Deliverables

### 1. `scripts/export_substack.py` (new; standard library + local Chrome)

Renders each visual component of `docs/index.html` to its own PNG. It never
modifies `docs/index.html`.

- **Image list:** a list of `(filename, CSS selector, alt text)` at the top of
  the file. It's the only thing to edit when components change.

  | File | Selector | Component |
  | --- | --- | --- |
  | `01-definitions.png` | `.defs` | Frontier LLM vs embedding model cards |
  | `02-candidates.png` | `.table-wrap:has(.conditions)` | Method table with tags |
  | `03-how-they-choose.png` | `.how` | The two "How … chooses" cards |
  | `04-dataset-facts.png` | `.facts` | Source, posts, seed grid |
  | `05-labels.png` | `#dataset .table-wrap` | Labels with descriptions |
  | `06-prices.png` | `.chart` | List price bar chart |
  | `07-results.png` | `.table-wrap:has(.results)` | Results table (placeholder until filled) |

- **Per image:**
  1. Write a temporary copy of the page with `data-theme="light"` on `<html>`,
     a white body background so it blends into Substack, and an injected
     script. The script replaces `.page`'s children with just the selected
     element. Once `document.fonts.ready` resolves, it records the content
     height and the number of matched elements as attributes on `<html>`.
  2. Run `--dump-dom` in headless Chrome with `--virtual-time-budget` to read
     the height and count. **Fail loudly if the count is 0**, so a renamed
     class can't silently produce an empty image.
  3. Take a headless Chrome `--screenshot` at `--window-size=WIDTH,height`
     with `--force-device-scale-factor=2`.
- **`WIDTH = 600` CSS px** (PNG width 1200px). This is below the page's
  680px breakpoint, so cards and tables use the stacked phone layout and stay
  readable when Substack scales images down on phones.
- **Chrome path:** `/Applications/Google Chrome.app/...`, overridable with a
  `CHROME` environment variable.
- **Run:** `uv run python scripts/export_substack.py`. It writes to
  `substack/` and prints each file's pixel size.

The approach is the same one already used this session to render and
measure the page (headless Chrome with `--dump-dom`, `--screenshot` and
`--virtual-time-budget`).

### 2. `substack/draft.html` (new; written by hand, not generated)

Plain semantic HTML with no CSS. Open it in a browser, select all, copy, and
paste into the Substack editor. Rich-text paste keeps headings, bold, links,
lists and, ideally, code blocks. Reading order:

1. **Title and subtitle** as plain text for Substack's title fields:
   "AI Classifier: Embedding Model vs Frontier LLMs" and the page's meta
   description.
2. The intro paragraph, plus "Interactive version:
   xcollantes.github.io/categorize-research".
3. **[Image 1]** definitions, then the three key-point bullets.
4. **Research questions:** a native numbered list.
5. **Method:** one line of text, **[Image 2]** and **[Image 3]**.
6. **Dataset:** intro text, **[Image 4]**, **[Image 5]**, and the stripping
   note as a paragraph.
7. **Metrics and prices:** one line, **[Image 6]**, and the macro-F1 and
   paired-bootstrap note.
8. **Results:** **[Image 7]** and the "pending" line.
9. **Prompt:** three code blocks (LLM prompt with `{post text}`, reply
   format, embedding input), each with its one-line explanation.
10. **Limitations:** a native bullet list.
11. **Footer:** the source-code link.

**Image markers:** each marker is a `<figure>` with
`<img src="0N-….png" alt="…">` and a caption "Image N · drag
0N-….png here". The alt text in the HTML matches the script's alt text, so
it's ready for Substack's image description field.

**Dropped from the page:** the "Jump to a section" buttons (in-page anchors
don't carry over to Substack) and the byline (Substack shows its own).

### 3. `README.md`

Add `scripts/export_substack.py` and `substack/` to the Repository layout
table. Add one line to Reproduction: rerun the export after updating
`docs/index.html`.

## Known limitation (say it; don't build around it)

Local images in pasted HTML almost certainly won't upload into Substack.
Images are added by dragging each PNG onto its marker. Hosting the PNGs on
GitHub Pages so paste can pull them in is a possible later step if manual
placement gets tedious.

## Verification

1. `uv run python scripts/export_substack.py` produces 7 PNGs in
   `substack/`, each 1200px wide with a plausible height.
2. View every PNG:
   - light theme, white background
   - Archivo / IBM Plex fonts loaded, not fallbacks
   - nothing clipped at the right or bottom edge
   - only the intended component in each image
3. **Phone legibility:** downscale each PNG to 375px wide (`sips -Z`) and view
   it. Body text must still be readable. If not, lower `WIDTH` and rerun.
4. Temporarily point one selector at a nonexistent class and confirm the
   script exits with an error instead of writing a blank image. Then revert.
5. Screenshot `substack/draft.html` in headless Chrome. Check the reading
   order, and that every `<img src>` exists in `substack/`.
6. **User step (manual, never publish):** paste `draft.html` into a new
   Substack draft, drag in the images, and check that headings, lists, links
   and code blocks survived. Report anything Substack mangled, especially
   code blocks.
