# /data/rag/build_index.py
import os, sys, json, orjson, faiss, numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any
from FlagEmbedding import BGEM3FlagModel  # BGE-M3 (dense 1024차원)
import re

DATA_PATH = Path("/data/rag/insurance_corpus.jsonl")
INDEX_DIR = Path("/data/rag/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

DOCS_NPY = INDEX_DIR / "docs.npy"          # 텍스트 청크
META_JSON = INDEX_DIR / "metas.jsonl"      # 메타데이터(줄 단위)
VECS_NPY = INDEX_DIR / "embeddings.npy"    # 임베딩 행렬(float32)
FAISS_IDX = INDEX_DIR / "faiss.index"      # FAISS 인덱스

CHUNK_CHARS = 800     # 한국어 기준 600~1000자 권장
CHUNK_OVERLAP = 80

def read_jsonl(path: Path):
    with path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            yield orjson.loads(line)

def sent_split_korean(text: str) -> List[str]:
    # 간단한 문장 분리(마침표/물음표/느낌표/줄바꿈)
    parts = re.split(r'(?<=[\.\?\!])\s+|\n+', text)
    return [p.strip() for p in parts if p.strip()]

def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    sents = sent_split_korean(text)
    chunks = []
    buf = ""
    for s in sents:
        if len(buf) + 1 + len(s) <= chunk_chars:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                chunks.append(buf)
            # overlap 고려: 이전 청크의 끝부분 일부를 새 청크의 시작으로
            if overlap > 0 and len(buf) > overlap:
                prefix = buf[-overlap:]
            else:
                prefix = ""
            buf = (prefix + " " + s).strip()
    if buf:
        chunks.append(buf)
    # 너무 긴 단일 문장은 강제 슬라이싱
    out = []
    for c in chunks:
        if len(c) <= chunk_chars:
            out.append(c)
        else:
            for i in range(0, len(c), chunk_chars - overlap):
                out.append(c[i:i+chunk_chars])
    return out

def build_corpus():
    docs = []
    metas = []
    for obj in tqdm(read_jsonl(DATA_PATH), desc="read"):
        text = obj.get("page_content") or obj.get("content") or obj.get("text") or ""
        md = obj.get("metadata") or {}
        if not text.strip():
            continue
        for ch in chunk_text(text, CHUNK_CHARS, CHUNK_OVERLAP):
            docs.append(ch)
            metas.append({
                "source": md.get("source", ""),
                "page": md.get("page", None)
            })
    return docs, metas

def main():
    if not DATA_PATH.exists():
        print(f"ERR: {DATA_PATH} not found", file=sys.stderr)
        sys.exit(1)

    print("1) Loading & chunking...")
    docs, metas = build_corpus()
    print(f" - chunks: {len(docs)}")

    print("2) Embedding with BGE-M3 (dense, CPU by default)...")
    # CPU 권장(서빙시 GPU는 LLM에 집중). GPU 사용 시 model_kwargs={"device": "cuda"}
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)  # fp16=False (CPU)
    out = model.encode_corpus(docs, batch_size=64)  # returns dict
    dense_vecs = out["dense_vecs"].astype(np.float32)
    # L2 normalize for cosine ~ inner product
    norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True) + 1e-12
    dense_vecs = dense_vecs / norms

    print("3) Save arrays & metas...")
    np.save(DOCS_NPY, np.array(docs, dtype=object), allow_pickle=True)
    with META_JSON.open("wb") as f:
        for m in metas:
            f.write(orjson.dumps(m))
            f.write(b"\n")
    np.save(VECS_NPY, dense_vecs)

    print("4) Build FAISS index (IP)...")
    d = dense_vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(dense_vecs)
    faiss.write_index(index, str(FAISS_IDX))
    print("Done.")

if __name__ == "__main__":
    main()
