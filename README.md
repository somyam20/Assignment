import os
import streamlit as st
from groq import Groq
#from dotenv import load_dotenv
#load_dotenv()
groq_api_key = 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
if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please set it in the .env file.")
    st.stop()
client = Groq(api_key=groq_api_key)
st.set_page_config(page_title="Chatbot", layout="centered")
st.title("Chatbot")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        user_input = st.chat_input("Ask your Questions...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="llama-3-8b-instruct",  
                messages=st.session_state.chat_history
            )
            assistant_reply = response.choices[0].message.content
            st.markdown(assistant_reply)
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})


import streamlit as st
from groq import Groq

# Hardcoded API Key (LOCAL USE ONLY)
groq_api_key = "your_actual_groq_api_key_here"

if not groq_api_key:
    st.error("GROQ_API_KEY not found.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=groq_api_key)

# Streamlit UI setup
st.set_page_config(page_title="Chatbot", layout="centered")
st.title("Groq-powered Chatbot")

# Chat history init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask your question...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="llama-3-8b-instruct",
                messages=st.session_state.chat_history
            )
            assistant_reply = response.choices[0].message.content
            st.markdown(assistant_reply)

    st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})