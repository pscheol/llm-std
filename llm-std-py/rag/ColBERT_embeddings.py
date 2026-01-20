from ragatouille import RAGPretrainedModel
import requests

RAG = RAGPretrainedModel.from_pretrained('colbert-ir/colbertv2.0')

def get_wikipedia_page(title: str):
    """
        위키백과의 페이지 호출
        :param title: str - 페이지 제목
        :return: str -페이지 내용을 row 문자열로 반환
    """
    #URL
    URL = "https://en.wikipedia.org/w/api.php"

    #요청 파라미터
    params = {
        "action":"query",
        "format":"json",
        "titles":title,
        "prop":"extracts",
        "explaintext":True,
    }

    #header 설정
    headers = {
        "User-Agent":"RAGatouille_std/0.0.1"
    }

    response = requests.get(URL, params=params, headers=headers)
    data = response.json()

    ## 페이지 컨텐츠 추출
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None

full_document = get_wikipedia_page("Herry Potter")

# 인덱스 설정
RAG.index(
    collection=[full_document],
    index_name="Herry-123",
    max_document_length=180,
    split_documents=True
)

# 쿼리
results = RAG.search(query="Who is Herry Potter", k=3)
print(f"Query={results}")
#랭체인에 전달
retriever = RAG.as_lengchain_retriever(k=3)
response = retriever.invoke("Who is Herry Potter")
print(f"response={response}")