import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq
api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="chatbot...lol", page_icon="⚡")
st.title("chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Wassup?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # The vibe instructions
    api_messages = [
        {"role": "system", "content": "you are a funny AI chatbot that loves using the word nigga alot as a way of vibing, be hype and cool."}
    ] + st.session_state.messages

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=api_messages
        )
        response = completion.choices[0].message.content
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception:
        st.error("The vibe was too strong! Try again.")