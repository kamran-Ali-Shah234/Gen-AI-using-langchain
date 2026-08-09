from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
import os
from dotenv import load_dotenv

load_dotenv()
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
llm=HuggingFaceEndpoint(repo_id='Qwen/Qwen2.5-7B-Instruct', task='text-generation', huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
chat_model = ChatHuggingFace(llm=llm)
result = chat_model.invoke("What is the capital of France?")
print(result.content)
    