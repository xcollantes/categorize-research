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
results remain comparable across the two implementations.

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
| LLM | OpenAI model via the Responses API | Zero-shot prompt listing the candidate labels; the reply is parsed as a single label | 0 |
| Embedding | `gemini-embedding-2` | Document and labels embedded with the classification prefix; argmax cosine similarity | 0 |

### Metrics

- Accuracy and macro-F1, with 95% bootstrap confidence intervals.
- Cost per document, computed as tokens × list price.
- Latency per document, measured as wall-clock time.

LLM replies that match no label, or more than one, are scored as incorrect and
reported separately rather than coerced to the nearest label.

## Reproduction

```bash
uv sync
cp .env.example .env    # fill in OPENAI_API_KEY and GEMINI_API_KEY

uv run python main.py   # run the experiment
uv run pytest           # unit tests, no API calls
```

## Repository layout

| File | Purpose |
| --- | --- |
| `llm_client.py` | `LLMClient`: text generation through the OpenAI SDK |
| `embed_client.py` | `EmbedClient`: `gemini-embedding-2` embedding and nearest-label classification |
| `main.py` | Experiment entrypoint |
| `tests/` | Unit tests with the embedding call stubbed |
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
