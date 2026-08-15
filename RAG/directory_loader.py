from langchain_community.document_loaders import DirectoryLoader , TextLoader
loader= DirectoryLoader(
    path="aaa",
    glob="*.txt",
    loader_cls=TextLoader
) 
docs=loader.load()
print(docs[0].page_content)