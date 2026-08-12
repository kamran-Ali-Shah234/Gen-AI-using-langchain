from langchain_huggingface import ChatHuggingFace
from huggingface_hub import  InferenceClient
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
llm=HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),temperature=0.2)
model=ChatHuggingFace(llm=llm)
#first prompt
template1=PromptTemplate(template="explain the following topic in detail:{topic}",
                        input_variables=["topic"])
#second prompt
template2=PromptTemplate(template= "write 5 line summary of the following text:{text}",
                        input_variables=["text"])
parser=StrOutputParser()
chain=template1 | model |parser | template2 | model | parser
result=chain.invoke({"topic":"Artificial Intelligence"})
print(result)