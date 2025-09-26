import json
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 데이터 및 저장 경로 설정
JSONL_PATH = "output/insurance_corpus.jsonl"
FAISS_INDEX_PATH = "vector_store/faiss_index"

def load_docs_from_jsonl(file_path):
    """JSONL 파일에서 문서를 로드하여 LangChain Document 객체 리스트로 반환합니다."""
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            doc = Document(page_content=data['page_content'], metadata=data['metadata'])
            documents.append(doc)
    return documents

def create_and_save_faiss_index():
    """
    문서를 로드하고, 임베딩하여 FAISS 벡터 스토어를 생성한 후,
    로컬 디스크에 저장합니다.
    """
    # 1. JSONL 파일에서 문서 로드
    documents = load_docs_from_jsonl(JSONL_PATH)
    if not documents:
        print("문서를 로드하지 못했습니다. JSONL 파일이 비어있는지 확인하세요.")
        return
    print(f"'{JSONL_PATH}'에서 {len(documents)}개의 문서를 로드했습니다.")

    # 2. 임베딩 모델 초기화
    # 한국어 특화 임베딩 모델을 사용합니다.
    # 'device':'cuda' 설정을 통해 GPU를 사용하여 임베딩 속도를 높입니다.
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print(f"임베딩 모델 '{model_name}'을(를) 초기화했습니다.")

    # 3. FAISS 벡터 스토어 생성
    # from_documents 메서드는 문서 리스트를 받아 임베딩을 수행하고
    # FAISS 인덱스를 내부적으로 생성합니다.
    print("FAISS 벡터 스토어 생성을 시작합니다. 문서 수에 따라 시간이 소요될 수 있습니다...")
    vector_store = FAISS.from_documents(documents, embeddings)
    print("FAISS 벡터 스토어 생성이 완료되었습니다.")

    # 4. 생성된 인덱스를 로컬에 저장
    # save_local 메서드를 사용하면 나중에 빠르게 로드하여 재사용할 수 있습니다.
    vector_store.save_local(FAISS_INDEX_PATH)
    print(f"FAISS 인덱스를 '{FAISS_INDEX_PATH}' 경로에 저장했습니다.")

if __name__ == "__main__":
    create_and_save_faiss_index()
