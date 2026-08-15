from langchain_huggingface import ChatHuggingFace
from huggingface_hub import  InferenceClient
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.schema.runnable import RunnableParallel,RunnableSequence
from dotenv import load_dotenv
import os
load_dotenv()
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
llm=HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),temperature=0.2)
model1=ChatHuggingFace(llm=llm)
model2=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite",task="text-generation",temperature=0.7)
prompt1=PromptTemplate(
    template="create notes from the following text:{text}",
    input_variables=["text"]
)
prompt2=PromptTemplate(
    template="write a 5 short question quiz from following text:{text}",
    input_variables=["text"]
)
prompt3=PromptTemplate(
    temlate="merge the provided notes and quiz into a single document \n notes:{notes} and {quiz} ",
    input_variables=["notes","quiz"]
)
parser=StrOutputParser()
parallelchain=RunnableParallel({
    "chain": RunnableSequence(prompt1,model1,parser),
    "chain2": RunnableSequence(prompt2,model2,parser)
}
)
mergechain=RunnableSequence(prompt3,model1,parser)
finalchain=RunnableSequence(parallelchain,mergechain)
text=""
result=finalchain.invoke({"text":text})