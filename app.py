import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- 1. 전역 변수 및 모델 로딩 ---

# FastAPI 앱 초기화
app = FastAPI()

# 모델 및 벡터 스토어 로드를 위한 전역 변수
llm_pipeline = None
vector_store = None
qa_chain = None

# Pydantic 모델 정의 (API 요청 본문)
class QueryRequest(BaseModel):
    query: str

# --- 2. 모델 및 벡터 스토어 로딩 함수 ---

def load_models():
    """
    서버 시작 시 LLM, 임베딩 모델, 벡터 스토어를 로드하여 전역 변수에 할당합니다.
    이 함수는 서버 시작 시 한 번만 호출됩니다.
    """
    global llm_pipeline, vector_store, qa_chain

    # --- GPU 사용 가능 여부 확인 ---
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This application requires a GPU.")
    
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # --- 임베딩 모델 로드 ---
    print("Loading embedding model...")
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # --- FAISS 벡터 스토어 로드 ---
    print("Loading FAISS vector store...")
    FAISS_INDEX_PATH = "vector_store/faiss_index"
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Vector store loaded successfully.")

    # --- gpt-oss-20b LLM 로드 ---
    print("Loading gpt-oss-20b model...")
    llm_model_id = "openai/gpt-oss-20b" # 실제 모델 ID로 변경 필요 시
    tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    # 4-bit 양자화를 사용하여 VRAM 사용량 감소 (하드웨어 사양에 따라 조정)
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_id,
        torch_dtype="auto",
        device_map="auto"
    )
    
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )
    hf_pipeline = HuggingFacePipeline(pipeline=llm_pipeline)
    print("gpt-oss-20b model loaded successfully.")

    # --- RetrievalQA 체인 설정 ---
    retriever = vector_store.as_retriever(search_kwargs={'k': 3}) # 상위 3개 문서 검색

    # 프롬프트 템플릿 정의: LLM에게 역할을 부여하고, 컨텍스트 기반 답변을 지시
    prompt_template = """
    주어진 컨텍스트 정보를 사용하여 다음 질문에 답변해 주세요. 컨텍스트에서 답을 찾을 수 없다면, "정보를 찾을 수 없습니다."라고 답변하세요. 답변은 한국어로, 간결하게 3문장 이내로 작성해 주세요.

    컨텍스트:
    {context}

    질문: {question}

    답변:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=hf_pipeline,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    print("RetrievalQA chain is ready.")

# --- 3. FastAPI 이벤트 핸들러 및 엔드포인트 ---

@app.on_event("startup")
async def startup_event():
    """FastAPI 서버 시작 시 모델을 로드합니다."""
    print("Server startup: Loading models...")
    load_models()
    print("Models loaded. Server is ready.")

@app.post("/ask")
async def ask_query(request: QueryRequest):
    """
    사용자의 질문을 받아 RAG 체인을 통해 답변을 생성하고 반환합니다.
    """
    if not qa_chain:
        raise HTTPException(status_code=503, detail="QA chain is not initialized.")

    try:
        query = request.query
        print(f"Received query: {query}")
        
        # RAG 체인 실행
        result = qa_chain.invoke({"query": query})
        
        # 결과 포맷팅
        answer = result.get("result", "No answer found.")
        source_documents = result.get("source_documents",)
        
        sources = []
        for doc in source_documents:
            sources.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        return {
            "query": query,
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        print(f"Error during query processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

if __name__ == "__main__":
    import uvicorn
    # 서버 실행 (개발용)
    uvicorn.run(app, host="0.0.0.0", port=8000)
