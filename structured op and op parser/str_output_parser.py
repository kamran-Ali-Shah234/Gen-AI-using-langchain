from langchain_huggingface import ChatHuggingFace
from huggingface_hub import  InferenceClient
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
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
prompt1=template1.format(topic="Artificial Intelligence")
result=model.invoke(prompt1)
prompt2=template2.format(text=result.content)
result2=model.invoke(prompt2)
print("Summary:", result2.content)