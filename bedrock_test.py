from langchain_aws import Bedrock
from langchain.chains.llm import LLMChain
import streamlit as st
from langchain.prompts import PromptTemplate
import boto3

bedrock_client = boto3.client(
    service_name='bedrock_runtime',
    region_name='us-east-1'
)


model_id = ""

llm=Bedrock(
    bedrock_client,
    model_id,
    model_kwargs={'temperature':0.9}
)

def chatbot(language,user_text):
    prompt = PromptTemplate(input_variables=["language","user_text"],
                            template="You are a chatbot , you are in {language} \n\n{user_text}")

    bedrock_chain = LLMChain(llm=llm, prompt= prompt)
    response=bedrock_chain.invoke({'language':language,'user_text':user_text})
    return response

st.title("Bedrock chatbot demo")
language = st.sidebar.selectbox(label="Language",options=["english", "spanish","vietnam"])

if language:
    user_text = st.sidebar.text_area(label="What is your questions ?",max_chars=100)

if user_text:
    response = chatbot(language,user_text)

    st.write(response['text'])
