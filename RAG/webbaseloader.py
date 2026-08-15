from langchain_community.document_loaders import WebBaseLoader
url="https://www.kaggle.com"
loader=WebBaseLoader(url)
docs=loader.load()
print(docs)