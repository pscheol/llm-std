from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter
)

markdownText = '''
# Hello world!! 
# LangChain Build applications with LLMs through composability

## Quick Install
\'\'\'bash
pip install langchain
\'\'\'

'''

markdown_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN, chunk_size=60, chunk_overlap=0)
markdown_docs = markdown_splitter.create_documents([markdownText])

print(markdown_docs)