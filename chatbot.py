import streamlit as st
from utils import Chatbot

st.set_page_config(page_title="Chatbot", page_icon="🤖")

system_prompt = st.sidebar.text_area("System Prompt", value="""
You are a polite chatbot that answers concisely and truthfully.
""".strip())

bot = Chatbot('open-mixtral-8x7b', system_prompt=system_prompt)

if st.sidebar.button("Reset conversation"):
    bot.reset()

for message in bot.history():
    with st.chat_message(message.role):
        st.write(message.content)

msg = st.chat_input()

if not msg:
    st.stop()

with st.chat_message("user"):
    st.write(msg)

with st.chat_message("assistant"):
    st.write_stream(bot.submit(msg, context=5))
