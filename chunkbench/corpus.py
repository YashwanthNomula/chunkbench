"""Built-in benchmark corpus: production-ML documents with labeled questions.

Each question names a ``gold_doc`` and one or more ``must_contain`` phrases
that appear verbatim in that document. A retrieval hit requires a returned
chunk to come from the gold document AND contain every phrase, so chunking
that splits an answer across a boundary is directly punished.
"""
from __future__ import annotations

import json
from pathlib import Path

DOCS = [
    {
        "id": "lora",
        "title": "LoRA: Low-Rank Adaptation for Fine-Tuning",
        "text": """LoRA (Low-Rank Adaptation) fine-tunes large models by freezing the
pretrained weights and injecting trainable rank decomposition matrices into
each layer. Instead of updating a weight matrix W of shape d by k directly,
LoRA learns two small matrices A and B whose product approximates the update,
so the number of trainable parameters drops by orders of magnitude.

Two hyperparameters control the adapter. The rank r sets the capacity of the
update: r=8 is a common default, while r=64 captures more complex tasks at
higher memory cost. Alpha scales the update through a scaling factor of alpha
divided by r, which keeps the effective learning rate stable when you change
the rank. A typical starting point is r=16 with alpha=32.

In practice you choose target modules such as q_proj and v_proj for the
attention projections, and optionally add k_proj, o_proj, and the MLP gates
for harder tasks. Training uses the same optimizer and schedule as full
fine-tuning, but only the adapter weights receive gradients, so a 7B model
can be adapted on a single consumer GPU.

At inference time you can merge the adapter weights back into the base model
for zero-latency serving, or keep adapters separate and hot-swap them per
request when one base model serves many tenants. Full fine-tuning still wins
when the task requires deep changes to the model's knowledge, but for
instruction following and style adaptation LoRA matches it closely.""",
    },
    {
        "id": "rag-eval",
        "title": "Evaluating RAG Systems",
        "text": """A RAG system fails in two places: retrieval and generation. Retrieval is
measured with rank-aware metrics like recall at k and MRR against labeled
question-document pairs. Generation needs its own metrics because a fluent
answer can still be wrong.

Faithfulness measures whether every claim in the answer is supported by the
retrieved context. It is the single most important generation metric for
production RAG: an unfaithful answer is a hallucination even when it sounds
confident. Answer relevancy then checks whether the response actually
addresses the user's question rather than drifting into related trivia.

Context precision and context recall complete the picture from the retrieval
side. Context recall asks whether all the information needed to answer is
present in the retrieved chunks, which makes it sensitive to chunking
strategy: split an answer across a chunk boundary and recall drops even
though the text is technically in the index.

Teams usually start with an LLM-as-judge pipeline that scores faithfulness
and relevancy on a sample of production traffic, then graduate a subset of
cases to human review for calibration. Judge prompts must be versioned like
code, because a changed judge silently moves every metric. Finally, chunk
size directly trades off precision against recall: smaller chunks rank more
precisely but fracture multi-sentence answers.""",
    },
    {
        "id": "drift",
        "title": "Detecting Data Drift in Production ML",
        "text": """Data drift means the serving distribution has moved away from the
training distribution. The model may still return predictions with high
confidence while its accuracy quietly degrades, which is why drift
monitoring is a standard production requirement.

The Population Stability Index (PSI) is the workhorse metric for tabular
features. You bin the reference distribution, bin the current serving
window the same way, and sum (actual minus expected) times log(actual over
expected) across bins. A PSI above 0.25 is the conventional signal for
significant drift that deserves investigation, while values under 0.1 are
usually noise.

The Kolmogorov-Smirnov test complements PSI for continuous features: it
measures the maximum gap between the two cumulative distributions and ships
with a p-value, which makes it convenient for automated alerting. PSI is
more interpretable for business stakeholders; KS is more sensitive to
small shifts in the middle of the distribution.

A sound setup compares a fixed reference window, typically the training
data or the first stable month of serving, against rolling serving
windows of a day or a week. Alert on the metric, but always plot the two
distributions before acting: a drifting feature that the model barely
uses is a ticket, not an incident. Covariate drift in inputs is common;
label drift and concept drift are rarer but far more damaging.""",
    },
    {
        "id": "quantization",
        "title": "Quantization for LLM Inference",
        "text": """Quantization shrinks model weights from 16-bit floats to 8-bit or 4-bit
integers, cutting VRAM roughly in half or by three quarters. For a 70B
model this is the difference between eight GPUs and two, so quantization
is often what makes self-hosting economically possible.

GPTQ uses second-order information from a small calibration set to decide
which weights can be rounded aggressively and which must stay precise. AWQ
takes a different route: it protects the small fraction of salient weights
identified by activation magnitudes and quantizes everything else. Both
outperform naive round-to-nearest by a wide margin on perplexity.

The cost shows up as perplexity degradation on held-out text and, more
importantly, on downstream task accuracy. Four-bit quantization of a 70B
model typically costs one to three points on MMLU-style benchmarks; 8-bit
is usually within noise of the full-precision baseline. Always measure on
your own eval set rather than trusting published numbers.

Two practical details matter. First, the KV cache also consumes memory at
long contexts and can be quantized separately from the weights. Second,
quantized training (QLoRA) and quantized inference compose: you can
fine-tune adapters on a 4-bit base model and serve the merged result in
8-bit. Calibration data should resemble your real prompts, because GPTQ
tuned on web text can underperform on code.""",
    },
    {
        "id": "vector-db",
        "title": "Vector Databases and Hybrid Retrieval",
        "text": """Vector databases store embedding vectors and answer nearest-neighbor
queries over millions of items in milliseconds. The standard index is
HNSW, a hierarchical small-world graph that trades a little recall for
dramatic speed: queries traverse from a coarse top layer down to a dense
bottom layer, visiting only a fraction of the vectors.

Pure dense retrieval misses exact terminology. A query for "PSI above 0.25"
may retrieve documents about p-values instead of the drift runbook that
contains the literal string. Hybrid retrieval fixes this by running BM25
keyword search alongside vector search and merging the two ranked lists
with reciprocal rank fusion, which needs no score normalization because
it works on ranks alone.

Metadata filtering turns the database into a real retrieval layer: filter
by tenant, document type, or freshness before the vector search runs, so
one tenant never sees another's chunks. Pre-filtering beats post-filtering
because the ANN index only explores the allowed subspace.

Chunk size is the lever with the biggest effect on quality. Small chunks
give precise matches but lose surrounding context; large chunks preserve
context but dilute the embedding with unrelated text. Most production
systems land between 300 and 800 tokens and re-rank the top candidates
with a cross-encoder before generation.""",
    },
    {
        "id": "prompt-cache",
        "title": "Prompt Caching and LLM Cost Control",
        "text": """LLM spend is dominated by input tokens on repeated prefixes: system
prompts, few-shot examples, and long retrieved contexts that barely change
between calls. Prompt caching attacks exactly this waste.

Prefix caching stores the key-value states for the start of a prompt, so a
repeat request with the same prefix skips recomputation. Providers implement
it automatically when the prefix is byte-identical, which rewards stable
prompt templates: put the variable user content at the end and keep
everything else frozen. A cache hit typically costs a tenth of a full
prefill and returns much faster.

The metric to watch is the cache hit rate, the fraction of input tokens
served from cache. Ninety percent is achievable for agents that loop over
the same context, while one-shot chatbots may see under twenty. Track it
per endpoint, because one uncached template can dominate the bill.

Cached entries expire: every provider sets a time to live, often five to
sixty minutes, after which the prefix is recomputed. Design refresh loops
so hot prefixes get touched before they expire. For budgeting, convert
everything to dollars per million tokens blended across cached and
uncached traffic, then set alerts on the blended rate rather than on raw
token counts.""",
    },
    {
        "id": "feature-store",
        "title": "Feature Stores for Machine Learning",
        "text": """A feature store is the shared layer between training and serving. It
stores versioned feature definitions, materializes them on a schedule, and
serves point lookups at prediction time, so the model always sees features
computed the same way they were computed in training.

The design splits into two stores. The offline store holds historical
feature values for training, usually in columnar files or a warehouse. The
online store holds only the latest values in a low-latency key-value
system for real-time inference. A single transformation definition feeds
both, which is what kills most training-serving skew.

Point-in-time correctness is the hard requirement. When you build a
training row labeled as of March 3rd, every feature must reflect values
known on March 3rd, not values backfilled later. Feature stores enforce
this with event timestamps and as-of joins; getting it wrong leaks the
future into training and inflates offline metrics that collapse online.

Adoption pays off through reuse: the churn features built for one model
become available to the next team without a new pipeline. Start with a
small set of widely used entity keys, such as user id and item id, and
expand only when a second consumer appears. Unused features are a
maintenance liability, not an asset.""",
    },
    {
        "id": "deploy",
        "title": "Deployment Gates for ML Models",
        "text": """A deployment gate is a checklist that must pass before a model artifact
is promoted to production. It turns "the notebook ran" into auditable
evidence: required files present, input schema matches, metrics beat the
baseline, latency fits the budget, and a prediction contract holds on a
canary sample.

Rollout strategy is part of the gate. A canary deployment sends a small
fraction of live traffic to the new model while the old one serves the
rest, so regressions show up in business metrics before they affect
everyone. A shadow deployment goes further: the new model scores every
request but its outputs are logged and discarded, which validates the full
serving path with zero user impact.

Every gate needs a rollback plan decided in advance. Define the metric and
threshold that triggers a rollback, who can approve it, and how traffic
shifts back. A canary without an automatic rollback is just a slow
incident.

In CI the gate runs as a job that exits non-zero on failure, blocking the
merge or the release. Gate definitions live in version control next to the
model code, and every promotion records which gate version ran, so an
incident review can always answer what was checked. The promotion gate is
the last line of defense between an experiment and your users.""",
    },
]

QUESTIONS = [
    {
        "text": "In LoRA, what does the alpha hyperparameter control?",
        "gold_doc": "lora",
        "must_contain": ["scaling factor of alpha divided by r"],
    },
    {
        "text": "Which modules are typically targeted by LoRA adapters?",
        "gold_doc": "lora",
        "must_contain": ["target modules such as q_proj and v_proj"],
    },
    {
        "text": "How can LoRA adapters be served with zero added latency?",
        "gold_doc": "lora",
        "must_contain": ["merge the adapter weights back into the base model"],
    },
    {
        "text": "What is the most important generation metric for production RAG?",
        "gold_doc": "rag-eval",
        "must_contain": ["faithfulness measures whether every claim"],
    },
    {
        "text": "Why does chunking strategy affect context recall?",
        "gold_doc": "rag-eval",
        "must_contain": ["split an answer across a chunk boundary"],
    },
    {
        "text": "What PSI value signals significant drift?",
        "gold_doc": "drift",
        "must_contain": ["psi above 0.25", "significant drift"],
    },
    {
        "text": "How does the Kolmogorov-Smirnov test detect drift?",
        "gold_doc": "drift",
        "must_contain": ["maximum gap between the two cumulative distributions"],
    },
    {
        "text": "How does GPTQ decide which weights to quantize aggressively?",
        "gold_doc": "quantization",
        "must_contain": ["gptq uses second-order information"],
    },
    {
        "text": "What is the typical accuracy cost of 4-bit quantization on a 70B model?",
        "gold_doc": "quantization",
        "must_contain": ["one to three points"],
    },
    {
        "text": "What index do vector databases typically use for ANN search?",
        "gold_doc": "vector-db",
        "must_contain": ["hnsw", "hierarchical small-world graph"],
    },
    {
        "text": "How does reciprocal rank fusion combine BM25 and vector results?",
        "gold_doc": "vector-db",
        "must_contain": ["reciprocal rank fusion", "works on ranks alone"],
    },
    {
        "text": "Where should variable user content go to maximize prefix cache hits?",
        "gold_doc": "prompt-cache",
        "must_contain": ["put the variable user content at the end"],
    },
    {
        "text": "What cache hit rate is achievable for agents looping over the same context?",
        "gold_doc": "prompt-cache",
        "must_contain": ["ninety percent"],
    },
    {
        "text": "Why must feature stores enforce point-in-time correctness?",
        "gold_doc": "feature-store",
        "must_contain": ["point-in-time correctness", "leaks the future into training"],
    },
    {
        "text": "What is the difference between the offline and online feature stores?",
        "gold_doc": "feature-store",
        "must_contain": ["offline store", "online store"],
    },
    {
        "text": "What is the difference between canary and shadow deployments?",
        "gold_doc": "deploy",
        "must_contain": ["canary deployment", "shadow deployment"],
    },
    {
        "text": "What should every ML deployment gate define in advance?",
        "gold_doc": "deploy",
        "must_contain": ["rollback plan", "triggers a rollback"],
    },
    {
        "text": "Why do vector databases need metadata filtering?",
        "gold_doc": "vector-db",
        "must_contain": ["metadata filtering", "one tenant never sees another"],
    },
    {
        "text": "What reward structure makes prefix caching effective?",
        "gold_doc": "prompt-cache",
        "must_contain": ["time to live", "touched before they expire"],
    },
    {
        "text": "When does full fine-tuning still beat LoRA?",
        "gold_doc": "lora",
        "must_contain": ["deep changes to the model's knowledge"],
    },
]


def load_docs_dir(path: str | Path) -> list[dict]:
    """Load ``*.md`` files from a directory as documents (filename = doc id)."""
    docs = []
    for md in sorted(Path(path).glob("*.md")):
        docs.append({"id": md.stem, "title": md.stem, "text": md.read_text()})
    if not docs:
        raise ValueError(f"no .md files found in {path}")
    return docs


def load_questions_json(path: str | Path) -> list[dict]:
    """Load questions from JSON: [{text, gold_doc, must_contain: [...]}]."""
    items = json.loads(Path(path).read_text())
    for q in items:
        if not {"text", "gold_doc", "must_contain"} <= set(q):
            raise ValueError(f"question missing required keys: {q}")
    return items
