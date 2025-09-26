import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 입력 및 출력 경로 설정
PDF_PATH = "data/보험.pdf"
OUTPUT_PATH = "output/insurance_corpus.jsonl"

def preprocess_pdf_to_jsonl():
    """
    PDF 파일을 로드하여 텍스트를 청크로 분할하고,
    각 청크를 JSONL 파일의 한 줄로 저장합니다.
    """
    # 출력 디렉터리가 없으면 생성
    output_dir = os.path.dirname(OUTPUT_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. PDF 파일 로드
    # PyPDFLoader는 PDF를 페이지별로 나누어 Document 객체 리스트로 반환합니다.
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"'{PDF_PATH}'에서 {len(documents)}개의 페이지를 로드했습니다.")

    # 2. 텍스트 분할기(Text Splitter) 설정
    # RecursiveCharacterTextSplitter는 의미 단위가 깨지지 않도록
    # 문단 -> 문장 -> 단어 순으로 재귀적으로 텍스트를 분할합니다.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 각 청크의 최대 크기 (문자 수 기준)
        chunk_overlap=200, # 청크 간의 중복 문자 수
        length_function=len,
        is_separator_regex=False,
    )

    # 3. 문서 분할 및 JSONL 파일로 저장
    chunks_count = 0
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for doc in documents:
            # 페이지 내용을 기반으로 텍스트 청크 생성
            chunks = text_splitter.split_text(doc.page_content)
            
            for chunk_content in chunks:
                # 각 청크와 메타데이터를 JSON 객체로 구성
                # 메타데이터에는 원본 소스와 페이지 번호가 포함됩니다.
                data = {
                    "page_content": chunk_content,
                    "metadata": {
                        "source": doc.metadata.get("source", "N/A"),
                        "page": doc.metadata.get("page", -1)
                    }
                }
                # JSON 객체를 문자열로 변환하고 파일에 한 줄로 기록
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
                chunks_count += 1
    
    print(f"총 {chunks_count}개의 청크를 생성하여 '{OUTPUT_PATH}'에 저장했습니다.")

if __name__ == "__main__":
    preprocess_pdf_to_jsonl()
