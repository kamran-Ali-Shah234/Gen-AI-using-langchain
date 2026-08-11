from langchain_huggingface import ChatHuggingFace
from huggingface_hub import  InferenceClient
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser,ResponseSchema
from dotenv import load_dotenv
import os
load_dotenv()
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
llm=HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),temperature=0.2)
model=ChatHuggingFace(llm=llm)
#first prompt

schema=[
    ResponseSchema(name="fact 1", description="fact 1 of about the topic"),
    ResponseSchema(name="fact 2", description="fact 2 of about the topic"),
    ResponseSchema(name="fact 3", description="fact 3 of about the topic")
]
parser=StructuredOutputParser.from_response_schemas(schema)
template=PromptTemplate(template="gave 3 fact about:{topic} \n {formet_instruction}",
                        input_variables=["topic"],
                        partial_variables={"formet_instruction": parser.get_format_instructions()})
chain=template | model | parser
result=chain.invoke({"topic": "machine learning"})
print(result)