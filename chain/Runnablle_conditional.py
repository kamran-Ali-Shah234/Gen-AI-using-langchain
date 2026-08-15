from langchain_huggingface import ChatHuggingFace
from huggingface_hub import  InferenceClient
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda,RunnableSequence
from pydantic import BaseModel , Field
from typing import Literal
from dotenv import load_dotenv
import os
load_dotenv()
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
llm=HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),temperature=0.1)
model=ChatHuggingFace(llm=llm)
class Sentiment(BaseModel):
    sentiment: Literal["positive ","negative"] = Field(description="gave the sentiment of the feedback")
parser1=StrOutputParser()
parser2=PydanticOutputParser(pydantic_object=Sentiment)
prompt1=PromptTemplate(
    template="classify the sentiment of following feedback text  into postive or negative:\n {feedback} \n {formet_instruction}",
    input_variables=["feedback"],
    partial_variables={"formet_instruction" : parser2.get_format_instructions()}
)
prompt2=PromptTemplate(
    template="write an appropriate response to this positive feedback \n {feedback}",
    input_variables=["feedback"]
)
prompt3=PromptTemplate(
    template="write an appropriate response to this negative feedback \n {feedback}",
    input_variables=["feedback"]
)
classifier_chain=RunnableSequence(prompt1,model,parser2)
branch_chain = RunnableBranch(
    (lambda x:x.sentiment=="positive",RunnableSequence( prompt2,model,parser1)),
    (lambda x:x.sentiment=="negative",RunnableSequence( prompt3,model,parser1)),
    RunnableLambda(lambda x:" could find any sentiment.")
)
chain=RunnableSequence(classifier_chain,branch_chain)
result=chain.invoke({"feedback": " this is terribale phone . they created what a garbage."})
print(result)