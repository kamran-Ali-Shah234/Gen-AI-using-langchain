from langchain_huggingface import ChatHuggingFace
from huggingface_hub import  InferenceClient
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os
load_dotenv()
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
llm=HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),temperature=0.9)
model=ChatHuggingFace(llm=llm)
parser=JsonOutputParser()
#first prompt
templat=PromptTemplate(template="gave me a name,city,age and profession of a fictional character in a json:{formet_instruction}",
                        input_variables=[],
                        partial_variables={"formet_instruction": parser.get_format_instructions()}
)
chain=templat|model|parser
result=chain.invoke({})
"""prompt=templat.format()
result=model.invoke(prompt)
results=parser.parse(result.content)"""
print(result["name"])
print(type(result))
