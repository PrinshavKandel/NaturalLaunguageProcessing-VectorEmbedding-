# Word Vector Embedding Explorer

Visualize and compare word embeddings using a real neural language model. Enter any two English words or phrases to see their 384-dimensional vector representations, cosine similarity, and a 2D projection of the vector space.

## Requirements

- Python 3.8+
- pip

## Installation

```bash
pip install sentence-transformers matplotlib numpy
```

## Usage

```bash
python VectorEmbedding.py
```

The model (`all-MiniLM-L6-v2`) downloads ~90 MB on first run, then works offline.

## What It Shows

- **Embedding bar chart** — top 20 most significant dimensions for each word
- **Cosine similarity** — how related the two words are (-1 to +1), with plain-English interpretation
- **2D PCA scatter plot** — both words projected into 2D alongside 27 reference words for context
- **Dimension delta table** — the 12 dimensions that differ most between the two vectors

## Model

| Property | Value |
|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimensions | 384 |
| Vocabulary | Any English word, phrase, or sentence |
| Runs locally | Yes (no API key needed) |

## Example

```
Enter word 1: king
Enter word 2: queen

Cosine similarity : 0.8914
Interpretation    : very similar
Angle (degrees)   : 26.93
```
