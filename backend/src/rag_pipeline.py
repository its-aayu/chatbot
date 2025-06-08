from src.vector_store import load_or_create_vector_store
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from groq import Groq
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_key)

vector_store = load_or_create_vector_store()

retriever = vector_store.as_retriever(search_kwargs={"k": 40})

template = """
You are a helpful assistant. Use the following extracted context from the Constitution of India
to answer the question at the end. If you don't know the answer, say you don't know.

Context:
{context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)

def set_llm(user_prompt):
    response = client.chat.completions.create(
        model="compound-beta-mini",  
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.choices[0].message.content

def bot(query):
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    user_prompt = prompt.format(context=context, question=query)
    return set_llm(user_prompt)

if __name__ == "__main__":
    query = "What is the fundamental right to equality?"
    response = bot(query)
    print(response)
