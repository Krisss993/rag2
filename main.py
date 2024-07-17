import streamlit as st
from langchain_groq import ChatGroq
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

import os


#LLM and key loading function
def load_LLM(groq_api_key):
    """Logic for loading the chain you want to use should go here."""
    # Make sure your groq_api_key is set as an environment variable
    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key)
    return llm


#Page title and header
st.set_page_config(page_title="Ask from CSV File with FAQs about Napoleon")
st.header("Ask from CSV File with FAQs about Napoleon")


#Input Groq API Key
def get_groq_api_key():
    input_text = st.text_input(
        label="Groq API Key ",  
        placeholder="Ex: sk-2twmA8tfCb8un4...", 
        key="groq_api_key_input", 
        type="password")
    return input_text

groq_api_key = get_groq_api_key()


if groq_api_key:
    # embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb_file_path = "my_vecdtordb"

    def create_db():
        loader = CSVLoader(file_path='napoleon-faqs.csv', source_column="prompt")
        documents = loader.load()
        vectordb = Chroma.from_documents(documents, embedding, persist_directory="./my_vectordb")

        # Save vector database locally
        vectordb.persist()


    def execute_chain():
        # Load the vector database from the local folder
        loader = CSVLoader(file_path='napoleon-faqs.csv', source_column="prompt")
        documents = loader.load()
        vectordb = Chroma.from_documents(documents, embedding, persist_directory="./my_vectordb")
        vectordb.get()

        # Create a retriever for querying the vector database
        retriever = vectordb.as_retriever(score_threshold=0.7)

        template = """Given the following context and a question, generate an answer based on this context only.
        In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
        If the answer is not found in the context, respond "I don't know." Don't try to make up an answer.

        CONTEXT: {context}

        QUESTION: {question}"""

        prompt = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )
        
        llm = load_LLM(groq_api_key=groq_api_key)

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            input_key="query",
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        return chain


    if __name__ == "__main__":
        create_db()
        chain = execute_chain()


    btn = st.button("Private button: re-create database")
    if btn:
        create_db()

    question = st.text_input("Question: ")

    if question:
        chain = execute_chain()
        response = chain.invoke(question)

        st.header("Answer")
        st.write(response["result"])
