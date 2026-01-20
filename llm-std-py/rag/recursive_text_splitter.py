## 문서를 청크로 분할
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader('data/ateention.txt', encoding='utf-8')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitter_docs = splitter.split_documents(docs)

print(splitter_docs)


