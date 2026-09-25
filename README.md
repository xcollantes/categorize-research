# Classification with LLMs vs Embedding Models

## Abstract

Topic classification is commonly delegated to frontier LLMs through zero-shot
prompting. An embedding model offers a cheaper alternative: the document and
each candidate label are embedded, and the label with the highest cosine
similarity to the document is assigned. We ask whether this nearest-label
embedding classifier matches a prompted LLM in accuracy, and at what difference
in cost and latency. Neither method uses labelled training data, so the
comparison isolates the representation (generative reasoning vs. a fixed
embedding space) rather than any learned classifier.

## Research questions

1. **Accuracy:** Does a zero-shot embedding classifier achieve accuracy and
   macro-F1 comparable to a zero-shot prompted LLM on the same test set?
2. **Cost:** What is the cost per document?
3. **Latency** How long does a query return results for each method?

## Method

I fed a collection of news articles and a list of labels and have the candidate
choose one from a list. The dataset already has labels which we will use as the
expected results.

### Dataset

All conditions are scored on the same test set drawn from
[20 Newsgroups](http://qwone.com/~jason/20Newsgroups/), a collection of Usenet
posts from the early 1990s. Each post's newsgroup is its ground-truth label.
Four groups are used, seed 67. This matches the prior experiment so that
results remain comparable across the two implementations. The standard test
split has 1,573 posts in these groups; 45 are empty after stripping and are
dropped, leaving 1,528.

| Label | Description shown to both candidates |
| --- | --- |
| `comp.graphics` | Computer graphics: rendering, 3D modeling, ray tracing, image file formats, image processing, and graphics software and algorithms. |
| `rec.sport.baseball` | Baseball: teams, players, games, pitching and hitting, statistics, trades, and standings. |
| `sci.space` | Space and spaceflight: NASA, rocket launches, satellites, space stations, planetary missions, astronomy, and space policy. |
| `sci.electronics` | Electronics: circuit design, electronic components, soldering, power supplies, amplifiers, radio, and test equipment. |

Headers, footers (signatures) and quoted replies are stripped from every post.
Headers contain a `Newsgroups:` line that states the label outright, and
signatures and quotes let a classifier match on the author or thread instead
of the topic. Stripping leaves only the body text.

## LLM candidate

The conventional LLM will have the following dumped into the user context:

- Prompt instructions for choosing one of the list of categories
- List of categories
- Structured data model as output

## Embedding Model candidate

1. Embed the document. The model turns it into a fixed-length vector of numbers,
   and texts with similar meanings get vectors that point in similar directions.
2. Embed each label the same way, giving four vectors.
3. Score each label with cosine similarity against the document: the cosine of
   the angle between the two vectors. It's 1.0 for the same direction and near 0
   for unrelated text.
4. Pick the label with the highest score. That argmax is the "choice".

### Conditions

| Condition | Model | Procedure | Labelled data |
| --- | --- | --- | --- |
| GPT | `gpt-6-sol` via the Responses API | Zero-shot prompt listing the labels with descriptions; output constrained to one label by schema | 0 |
| Gemini | `gemini-3.1-pro-preview` | Same prompt and schema as GPT | 0 |
| Embedding | `gemini-embedding-2` | Document and label descriptions embedded with the classification prefix; argmax cosine similarity | 0 |

### Metrics

- Accuracy and macro-F1, with 95% bootstrap confidence intervals, and a paired
  bootstrap CI on each LLM's macro-F1 minus the embedding's.
- Cost per document, computed as tokens × list price. LLM output includes
  reasoning/thinking tokens, which are billed as output.
- Latency per document, measured as wall-clock time.

The LLM output schema allows only the four labels, so an LLM can fail to label
a post only by refusing. Refusals and API errors are scored as incorrect and
counted separately. Errors are left out of the cost and latency averages.

## Reproduction

```bash
uv sync
cp .env.example .env    # fill in OPENAI_API_KEY and GEMINI_API_KEY

uv run pytest                          # unit tests, no API calls
uv run python -m src.write_dataset     # write the test set to data/test.jsonl
uv run python -m src.main --limit 5    # smoke test: 5 posts, real API calls
uv run python -m src.main              # full run, from the repo root
```

`src.write_dataset` downloads, filters and shuffles (seed 67) the test set once
and writes it to `data/test.jsonl`. `src.main` reads that file, so every run
scores the same posts in the same order. A run appends one result line per post
to `data/results.jsonl` as it goes, so a crash keeps completed posts. Line *i* of `results.jsonl` is line *i* of
`data/test.jsonl`; `--limit` keeps a prefix of the same order.

## Repository layout

| File | Purpose |
| --- | --- |
| `src/clients/llm_clients.py` | `GPTClient`, `GeminiClient`: zero-shot classification with structured output |
| `src/clients/embed_client.py` | `EmbedClient`: `gemini-embedding-2` embedding and nearest-label classification |
| `src/models/response_models.py` | `Label`, `LABEL_DESCRIPTIONS`, `LLMResponse`, `Prediction` |
| `src/models/candidate.py` | `Candidate`: a condition's model ID and token prices |
| `src/utils/pull_data.py` | Load the test set; save it as JSON Lines |
| `src/utils/metrics.py` | Accuracy, macro-F1, bootstrap and paired-difference CIs |
| `src/write_dataset.py` | Standalone script: write the test set to `data/test.jsonl` |
| `src/main.py` | Experiment entrypoint |
| `tests/` | Metric unit tests, no API calls |
| `docs/index.html` | Public summary, served by GitHub Pages |

## Publication

A summary for a general audience is served at
<https://xcollantes.github.io/categorize-research/>. The page is a single static
file, `docs/index.html`; `docs/.nojekyll` causes GitHub to serve it as-is. To
enable publication, set Settings → Pages → Source to "Deploy from a branch",
branch `main`, folder `/docs`, then push.

## Limitations

- Nearest-label classification depends on how well the label strings themselves
  describe their classes; terse or ambiguous label names disadvantage the
  embedding condition. I have added a robust expected data model in the Pydantic
  structured response to assist the Embedding Model.
- LLM latency is measured over the public internet and carries network variance.
- List prices change; cost figures are valid only at the time of the run.
- "Works on my machine". Running program on my machine which may differ from
  another machine.
