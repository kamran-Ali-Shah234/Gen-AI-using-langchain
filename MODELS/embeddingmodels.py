from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
os.getenv("GOOGLE_API_KEY")
embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview",output_dimensionality='64')
result=embeddings.embed_query("Hello world")
print(result)