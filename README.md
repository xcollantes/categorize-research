# Classification with LLMs vs Embedding Models

## Abstract

Topic classification is commonly delegated to frontier LLMs through
zero-shot prompting. An embedding model offers a cheaper alternative: the
document and each candidate label are embedded, and the label with the
highest cosine similarity to the document is assigned. We ask whether this
nearest-label embedding classifier matches a prompted LLM in accuracy, and
at what difference in cost and latency. Neither method uses labelled
training data, so the comparison isolates the representation (generative
reasoning vs. a fixed embedding space) rather than any learned classifier.

## Research questions

1. **Accuracy:** Does a zero-shot embedding classifier achieve accuracy and
   macro-F1 comparable to a zero-shot prompted LLM on the same test set?
2. **Cost:** What is the cost per document and latency per document of
   each method?
3. **Task instruction:** Does the classification task prefix recommended for
   `gemini-embedding-2` (`task: classification | query: `) measurably
   improve accuracy over unprefixed input?

## Method

### Conditions

| Condition | Model | Procedure | Labelled data |
| --- | --- | --- | --- |
| LLM | OpenAI model via the Responses API | Zero-shot prompt listing the candidate labels; the reply is parsed as a single label | 0 |
| Embedding | `gemini-embedding-2` | Document and labels embedded with the classification prefix; argmax cosine similarity | 0 |

### Dataset

All conditions are scored on the same test set: 20 Newsgroups, four classes
(`comp.graphics`, `rec.sport.baseball`, `sci.space`, `sci.electronics`),
headers, footers and quotes stripped, seed 67. This matches the prior
experiment so that results remain comparable across the two
implementations.

### Metrics

- Accuracy and macro-F1, with 95% bootstrap confidence intervals.
- Cost per document, computed as tokens × list price.
- Latency per document, measured as wall-clock time.

LLM replies that match no label, or more than one, are scored as incorrect
and reported separately rather than coerced to the nearest label.

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
<https://xcollantes.github.io/categorize-research/>. The page is a single
static file, `docs/index.html`; `docs/.nojekyll` causes GitHub to serve it
as-is. To enable publication, set Settings → Pages → Source to "Deploy from
a branch", branch `main`, folder `/docs`, then push.

## Limitations

- Nearest-label classification depends on how well the label strings
  themselves describe their classes; terse or ambiguous label names
  disadvantage the embedding condition.
- LLM latency is measured over the public internet and carries network
  variance.
- List prices change; cost figures are valid only at the time of the run.
