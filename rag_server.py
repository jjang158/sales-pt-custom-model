# /data/rag/rag_server.py
import os, time, json, orjson, faiss, numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Body, Request, Header, HTTPException
from pydantic import BaseModel
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from llama_cpp import Llama
from dotenv import load_dotenv

load_dotenv()
# ---- 경로/환경 ----
MODEL_PATH = os.environ.get("MODEL_PATH", "/data/models/bllossom/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf")
INDEX_DIR = Path(os.environ.get("INDEX_DIR", "/data/rag/index"))
DOCS_NPY = INDEX_DIR / "docs.npy"
META_JSON = INDEX_DIR / "metas.jsonl"
VECS_NPY = INDEX_DIR / "embeddings.npy"
FAISS_IDX = INDEX_DIR / "faiss.index"
API_KEY = os.environ.get("RAG_API_KEY", "change-me")  # 간단 보호

# ---- 로드(시작 시 1회) ----
print("Loading FAISS & arrays...")
index = faiss.read_index(str(FAISS_IDX))
docs = np.load(DOCS_NPY, allow_pickle=True).tolist()
metas = [orjson.loads(line) for line in open(META_JSON, "rb")]
emb = np.load(VECS_NPY).astype(np.float32)

print("Loading Embedding model (BGE-M3)...")
embed_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)  # CPU

print("Loading Reranker (BGE reranker v2 m3)...")
rerank_tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
rerank_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
rerank_model.eval()  # CPU 사용

def rerank(query: str, candidates: List[Dict[str, Any]], top_n: int = 5):
    pairs = [(query, c["text"]) for c in candidates]
    with torch.no_grad():
        inputs = rerank_tok(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
        scores = rerank_model(**inputs).logits.squeeze(-1)  # (N)
        scores = torch.sigmoid(scores).cpu().numpy().tolist()
    for c, s in zip(candidates, scores):
        c["score"] = float(s)
    return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_n]

print("Loading LLM (llama.cpp, CUDA offload)...")
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,          # 가능한 모든 레이어를 GPU로 오프로딩
    n_ctx=8192,               # Bllossom 8B 컨텍스트 권장치(8K) :contentReference[oaicite:12]{index=12}
    n_threads=os.cpu_count(), # CPU 스레드
    verbose=False,
    chat_format="llama-3"     # Llama-3 채팅 템플릿
)

SYSTEM_PROMPT = """\
당신은 보험 전담 RAG 비서입니다. 한국어로 정확하고 근거(출처)를 명시해 답변하세요.
- 제공된 컨텍스트 바깥을 단정하지 말고, 모르면 '제공된 자료에서 확인되지 않는다'고 답하세요.
- 핵심 요점을 먼저 간결히 요약하고, 뒤에 근거 인용을 [출처: 파일, p.페이지] 형식으로 붙이세요.
"""

def embed_query(q: str) -> np.ndarray:
    out = embed_model.encode_queries([q])
    v = out["dense_vecs"][0].astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v

def search(q: str, top_k: int = 30):
    qv = embed_query(q)
    D, I = index.search(qv.reshape(1, -1), top_k)
    items = []
    for idx, dist in zip(I[0], D[0]):
        if idx < 0: 
            continue
        m = metas[idx]
        items.append({
            "text": docs[idx],
            "source": m.get("source", ""),
            "page": m.get("page", None),
            "score_vec": float(dist)
        })
    return items

def build_context(chunks: List[Dict[str, Any]]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        tag = f"[출처: {c['source']}, p.{c['page']}]" if c.get("source") else "[출처: 미상]"
        lines.append(f"### 문서 {i}\n{c['text']}\n{tag}\n")
    return "\n".join(lines)

def ask_llm(user_query: str, context: str, max_tokens: int = 512, temperature: float = 0.2):
    # Llama-3 chat format 메시지
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"# 질문\n{user_query}\n\n# 참고 문맥\n{context}\n\n위 컨텍스트만 근거로 답변하세요."}
    ]
    out = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<|eot_id|>"]
    )
    return out["choices"][0]["message"]["content"]

# ---- FastAPI ----
app = FastAPI(title="Insurance RAG API (Bllossom 8B + BGE-M3)", version="1.0")

class RagRequest(BaseModel):
    query: str
    top_k: int = 30
    final_k: int = 5
    max_tokens: int = 512
    temperature: float = 0.2

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/v1/rag/query")
def rag_query(req: RagRequest, request: Request, x_api_key: Optional[str] = Header(default=None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    t0 = time.time()
    client_ip = request.client.host
    # 1) 검색
    cand = search(req.query, top_k=req.top_k)
    # 2) 리랭크
    reranked = rerank(req.query, cand, top_n=req.final_k)
    # 3) 컨텍스트
    context = build_context(reranked)
    # 4) 생성
    answer = ask_llm(req.query, context, max_tokens=req.max_tokens, temperature=req.temperature)
    elapsed = int((time.time() - t0) * 1000)
    return {
        "answer": answer,
        "citations": [
            {"source": c["source"], "page": c["page"], "retrieval_score": c.get("score", None)}
            for c in reranked
        ],
        "client_ip_seen_by_server": client_ip,
        "latency_ms": elapsed
    }
