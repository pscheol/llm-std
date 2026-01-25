from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_teddynote.document_compressors import LLMChainExtractor
from langchain_text_splitters import CharacterTextSplitter

from config.load_sys import gemma_model


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"문서 {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )



loader = TextLoader('hello.text')

docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)

split_docs = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model='embeddinggemma:300m')

db = FAISS.from_documents(split_docs, embeddings)

# 검색 설정을 지정. mmr 검색 설정.
config = {
    "configurable": {
        "search_type": "mmr",
        "search_kwargs": {"k": 2, "fetch_k": 10, "lambda_mult": 0.6},
    }
}


retriever: VectorStoreRetriever = db.as_retriever(config=config)

docs = retriever.invoke("멀티모달인란")

print(docs)

llm = ChatOllama(model=gemma_model)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

compressed_docs = {
    compression_retriever.invoke("멀티모달은 어떻게 사용해")
}

# pretty_print_docs(compressed_docs)