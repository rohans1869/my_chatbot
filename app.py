import gradio as gr
from huggingface_hub import InferenceClient
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_together import TogetherEmbeddings
from langchain_community.llms import Together


#os
os.environ["TOGETHER_API_KEY"] = os.getenv("API_TOKEN")


#load
loader = TextLoader("Resume_data.txt")
documents = loader.load()

# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


vectorstore = FAISS.from_documents(docs,
     TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
)

retriever = vectorstore.as_retriever()

model = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.0,
    max_tokens=500,
    top_k=0,
    # together_api_key="..."
)


# Provide a template following the LLM's original chat template.
# template = """<s>[INST] answer from context only as a person. and always answer in short answer.
# answer for asked question only, if he greets greet back.
template = """<s>[INST] answer from context only as if person is responding (use i instead of you in response). and always answer in short answer.
answer for asked question only, if he greets greet back.

{context}

Question: {question} [/INST] 
"""
prompt = ChatPromptTemplate.from_template(template) 

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

def greet(query1,history):
  try:
    if len(query1) < 2:
      return "Ask your Question again"
    else:
      answer = chain.invoke(query1)
      return answer
  except:
    return "Hi"

# gradio
description = "This is a chatbot application based on the Mixtral-8x7B model. Simply type an input to get started with chatting.\n Note : Bot can generate random response sometimes"
examples = [["what is your contact number?"], ["where you are currently working?"]]

gr.ChatInterface(greet,title = "Chat with my Bot", description=description,examples=examples).launch(debug = True)