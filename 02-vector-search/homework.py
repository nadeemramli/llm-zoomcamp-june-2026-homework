"""Homework 2 — Vector Search (LLM Zoomcamp 2026).

Run from the repo root:  uv run python 02-vector-search/homework.py
Uses the ONNX Embedder + the model already downloaded under ./models.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # .../02-vector-search
sys.path.insert(0, str(HERE / "embed"))          # import embedder.py
from embedder import Embedder                     # noqa: E402

from gitsource import GithubRepositoryDataReader, chunk_documents  # noqa: E402
from minsearch import Index, VectorSearch         # noqa: E402

MODEL_PATH = HERE / "models" / "Xenova" / "all-MiniLM-L6-v2"
embedder = Embedder(path=str(MODEL_PATH))

Q1_QUERY = "How does approximate nearest neighbor search work?"

# ---------------------------------------------------------------- Q1
v = embedder.encode(Q1_QUERY)
print(f"Q1  vector length = {len(v)}")
print(f"Q1  v[0] = {v[0]:.4f}")

# ---------------------------------------------------------------- load data
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]
print(f"\nLoaded {len(documents)} documents")

# ---------------------------------------------------------------- Q2
page = next(d for d in documents
            if d["filename"] == "02-vector-search/lessons/07-sqlitesearch-vector.md")
u = embedder.encode(page["content"])
print(f"Q2  cosine similarity = {float(np.dot(v, u)):.4f}")

# ---------------------------------------------------------------- chunks + X
chunks = chunk_documents(documents, size=2000, step=1000)
print(f"\n{len(chunks)} chunks; keys = {sorted(chunks[0].keys())}")


def encode_all(texts, batch=32):
    out = []
    for i in range(0, len(texts), batch):
        out.append(embedder.encode_batch(texts[i:i + batch]))
    return np.vstack(out)


X = encode_all([c["content"] for c in chunks])
print(f"X shape = {X.shape}")

# ---------------------------------------------------------------- Q3
scores = X.dot(v)
best = int(np.argmax(scores))
print(f"Q3  highest-scoring chunk file = {chunks[best]['filename']}  (score={scores[best]:.4f})")

# ---------------------------------------------------------------- Q4
vindex = VectorSearch()
vindex.fit(X, chunks)
q4 = embedder.encode("What metric do we use to evaluate a search engine?")
q4_res = vindex.search(q4, num_results=5)
print(f"Q4  first result file = {q4_res[0]['filename']}")

# ---------------------------------------------------------------- Q5
tindex = Index(text_fields=["content"])
tindex.fit(chunks)
q5_text = "How do I store vectors in PostgreSQL?"
q5_v = embedder.encode(q5_text)
vec_top5 = vindex.search(q5_v, num_results=5)
txt_top5 = tindex.search(q5_text, num_results=5)
vec_files = [r["filename"] for r in vec_top5]
txt_files = [r["filename"] for r in txt_top5]
only_vec = [f for f in dict.fromkeys(vec_files) if f not in txt_files]
print("Q5  vector top5 :", vec_files)
print("Q5  text   top5 :", txt_files)
print("Q5  in vector but NOT text =", only_vec)


# ---------------------------------------------------------------- Q6
def rrf(result_lists, k=60, num_results=5):
    scores, docs = {}, {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


q6_text = "How do I give the model access to tools?"
q6_v = embedder.encode(q6_text)
for n in (5, 10):
    vr = vindex.search(q6_v, num_results=n)
    tr = tindex.search(q6_text, num_results=n)
    fused = rrf([vr, tr])
    print(f"Q6  (lists of {n}) ranked first after RRF = {fused[0]['filename']}")
