from src.vector_store import load_or_create_vector_store
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_key)

# Prompt template
template = """
You are a helpful assistant. Use the following extracted context from the Constitution of India
to answer the question at the end. If you don't know the answer, say you don't know.

Context:
{context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)

# 🧠 Lazy-load everything inside this function
def bot(query: str):
    # Only load when needed
    vector_store = load_or_create_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 40})

    # Retrieve relevant documents
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Format and send to Groq model
    user_prompt = prompt.format(context=context, question=query)
    response = client.chat.completions.create(
        model="compound-beta-mini",  
        messages=[{"role": "user", "content": user_prompt}],
    )

    return response.choices[0].message.content
 
def load_bot():
    return bot
