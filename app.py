import streamlit as st
from llm_agent import llm_agent
from graph_visualization import visual

st.title("Movie Chatbot")

st.sidebar.title("Select from ChatBot features")

selected_option = st.sidebar.selectbox(
    "Choose an option", ["LLM Agent", "Data Visualization"]
)

if selected_option == "LLM Agent":
    llm_agent()
elif selected_option == "Data Visualization":
    visual()
else:
    st.write("Please select an option")
