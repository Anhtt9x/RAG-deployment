import boto3
from langchain_aws import Bedrock
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_aws import BedrockEmbeddings
from langchain.vectorstores import FAISS

def get_docs():
    documents = DirectoryLoader(path='data',glob='*.txt',loader_cls=PyPDFLoader).load()

    text_spliter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)

    docs = text_spliter.split_documents(documents)

    return docs

bedrock_client = boto3.client(
    service_name='bedrock_runtime',
    region_name='us-east-1'
)


model_id = ""

embedding = BedrockEmbeddings(client=bedrock_client,model_id=model_id)



def get_vector_store(docs):
    doc_search = FAISS.from_documents(docs,embedding)

    doc_search.save_local('faiss_vector_store')


llm_bedrock =Bedrock(
    bedrock_client,
    model_id,
    model_kwargs={'max_gen_len':512}
)

prompt_template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context','question'])


def get_response_llm(llm,query, doc_search):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=doc_search.as_retriever(), chain_type_kwargs={"prompt": prompt},
        chain_type="stuff", return_source_code=True
    )

    response = qa_chain.invoke({'query':query})
    return response['result']

def main():
    st.set_page_config("RAG DEMO")
    st.header("End to End RAG Application")

    user_input = st.text_input("Ask a question")

    
    with st.sidebar:
        st.title("Update or Create Vector Database")

        if st.button("Vector Update"):
            with st.spinner("Processing..."):
                docs = get_docs()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Send"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_vector_store",embedding,allow_dangerous_deserialization=True)
            st.write(get_response_llm(llm=llm_bedrock ,doc_search=faiss_index,query=user_input))

        

if __name__ =="__main__":
    main()