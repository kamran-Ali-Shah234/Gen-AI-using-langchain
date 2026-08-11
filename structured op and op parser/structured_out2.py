from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel
model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite', temperature=0.2)
class review(BaseModel):
    summary: str
    sentiment: str
structure_model=model.with_structured_output(review)

result=structure_model.invoke("this mobile phone is very good and has a good camera. it is very good for taking photos and videos. it has a long battery life and is very fast. it is also very affordable and has a good design. overall, it is a great phone for the price.")
print(result)