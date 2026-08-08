from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv(r"C:\Users\Kamran Ali Shah\Desktop\langchain\MODELS\LLM\.env")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.5,
    max_output_tokens=1024
)

result = llm.invoke("What is machine learning?")

print(result.content)