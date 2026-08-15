from langchain_huggingface import ChatHuggingFace
from huggingface_hub import  InferenceClient
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_classic.schema.runnable import RunnableSequence
from dotenv import load_dotenv
import os
load_dotenv()
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
llm=HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),temperature=0.1)
model=ChatHuggingFace(llm=llm)
prompt=PromptTemplate(
    template="explain this topic:{topic}",
    input_variables=["topic"]
)
parser=StrOutputParser()
chain=RunnableSequence(prompt,model,parser)
result=chain.invoke({"topic":"machine learning"})
print(result)