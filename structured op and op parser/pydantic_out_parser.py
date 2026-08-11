from langchain_huggingface import ChatHuggingFace
from huggingface_hub import  InferenceClient
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from dotenv import load_dotenv
import os
load_dotenv()
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
llm=HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),temperature=0.2)
model=ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name : str = Field(description="name of the person.")
    age : int = Field(gt=18)
    city : str = Field(description="city of person.")
parser=PydanticOutputParser(pydantic_object=Person)
template= PromptTemplate(
    template="generate an adult comic character from {place}; age must be greater than 18 \n {formet_instruction}",
    input_variables=["place"],
    partial_variables={"formet_instruction": parser.get_format_instructions()}
)
chain= template | model |parser
result= chain.invoke({"place": "england"})
print(result)