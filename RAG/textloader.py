from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace
from huggingface_hub import  InferenceClient
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
llm=HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),temperature=0.1)
model=ChatHuggingFace(llm=llm)
loader=TextLoader("cricket.txt",encoding="utf-8")
docs=loader.load()
prompt1=PromptTemplate(
    template="write a summary for the following doc:{doc}",
    input_variables=["doc"]
)
parser=StrOutputParser()
chain=prompt1|model|parser
result=chain.invoke({"doc": docs[0].page_content})
print(result)