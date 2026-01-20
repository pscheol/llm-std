from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('data/sample.pdf')

pages = loader.load()

print(pages)