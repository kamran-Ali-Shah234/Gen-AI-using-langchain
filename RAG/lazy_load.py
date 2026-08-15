from langchain_community.document_loaders import DirectoryLoader , TextLoader
loader= DirectoryLoader(
    path="aaa",
    glob="*.txt",
    loader_cls=TextLoader
) 
docs=loader.lazy_load()
print(docs)