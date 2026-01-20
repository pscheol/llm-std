'''
##############################################################################
RAG(Retrieval-Augmented Generation) 파이프라인은 기존의 언어 모델에 검색 기능을 추가하여,
주어진 질문이나 문제에 대해 더 정확하고 풍부한 정보를 기반으로 답변을 생성할 수 있게 해준다.
이 파이프라인은 크게 데이터 로드, 텍스트 분할, 인덱싱, 검색, 생성의 다섯 단계로 구성.

##############################################################################

1. 데이터 로드(Load Data)
RAG에 사용할 데이터를 불러오는 단계.
외부 데이터 소스에서 정보를 수집하고, 필요한 형식으로 변환하여 시스템에 로드.
예) 공개 데이터셋, 웹 크롤링을 통해 얻은 데이터, 또는 사전에 정리된 자료일 수 있다.
가져온 데이터는 검색에 사용될 지식이나 정보를 담고 있어야 한다.

langchain_community.document_loaders 모듈에서 WebBaseLoader 클래스를 사용하여
특정 웹페이지(위키피디아 정책과 지침)의 데이터를 가져오는 방법을 보여준다.
웹 크롤링을 통해 웹페이지의 텍스트 데이터를 추출하여 Document 객체의 리스트로 변환.
'''
# Data Loader - 웹페이지 데이터 가져오기
from langchain_community.document_loaders import WebBaseLoader

# 위키피디아 정책과 지침
url = 'https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EC%A0%95%EC%B1%85%EA%B3%BC_%EC%A7%80%EC%B9%A8'
loader = WebBaseLoader(url)

# 웹페이지 텍스트 -> Documents
docs = loader.load()

print(len(docs))
print(len(docs[0].page_content))
print(docs[0].page_content[5000:6000])

##############################################################################

'''
2. 텍스트 분할(Text Split)
불러온 데이터를 작은 크기의 단위(chunk)로 분할하는 과정.
자연어 처리(NLP) 기술을 활용하여 큰 문서를 처리가 쉽도록 문단, 문장 또는 구 단위로 나누는 작업으로 검색 효율성을 높이기 위한 중요한 과정이다.

RecursiveCharacterTextSplitter라는 텍스트 분할 도구를 사용하고 있다. 
간략하게 설명하면 12552 개의 문자로 이루어진 긴 문장을 최대 1000글자 단위로 분할한다.
200글자는 각 분할마다 겹치게 하여 문맥이 잘려나가지 않고 유지되게 한다. 실행 결과를 보면 18개 조각으로 나눠지게 된다..
LLM 모델이나 API의 입력 크기에 대한 제한이 있기 때문에,
제한에 걸리지 않도록 적정한 크기로 텍스트의 길이를 줄일 필요가 있다.
그리고, 프롬프트가 지나치게 길어질 경우 중요한 정보가 상대적으로 희석되는 문제가 있을 수도 있다.
따라서, 적정한 크기로 텍스트를 분할하는 과정이 필요하다.
'''
# Text Split (Documents -> small chunks: Documents)
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print(len(splits))
print(splits[10])
#page_content에는 분할된 텍스트 조각이 있다.
print(splits[10].page_content)
# metadata 속성을 통해 원본 문서의 정보를 포함하는 메타데이터를 출력.
print(splits[10].metadata)