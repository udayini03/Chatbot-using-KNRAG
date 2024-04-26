import streamlit as st
from lida import Manager, TextGenerationConfig, llm
from dotenv import load_dotenv
import os
from PIL import Image
from io import BytesIO
import base64

def visual():
   
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = "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"

    def base64_to_image(base64_string):
        byte_data = base64.b64decode(base64_string)
        return Image.open(BytesIO(byte_data))

    lida = Manager(text_gen=llm("openai"))
    textgen_config = TextGenerationConfig(
        n=1, temperature=0.5, model="gpt-3.5-turbo-0301", use_cache=True)

    # Use Streamlit session state to store history persistently
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    history = st.session_state['history']


    def update_history(query, image):
        history.append({"query": query, "image": image})


    def show_history():
        if history:
            st.header("History")
            for entry in history:
                st.subheader("Query")
                st.write(entry["query"])
                st.subheader("Visualization")
                st.image(entry["image"])
        else:
            st.write("No past queries found.")


    st.subheader("Query your Data to Generate Graph")
    file_uploader = st.file_uploader("Upload your CSV", type="csv")

    if file_uploader is not None:
        path_to_save = "filename1.csv"
        with open(path_to_save, "wb") as f:
            f.write(file_uploader.getvalue())

        text_area = st.text_area(
            "Query your Data to Generate Graph", height=200)

        if st.button("Generate Graph"):
            if len(text_area) > 0:
                st.info("Your Query: " + text_area)

                summary = lida.summarize(
                    "filename1.csv", summary_method="default", textgen_config=textgen_config)
                user_query = text_area
                charts = lida.visualize(
                    summary=summary, goal=user_query, textgen_config=textgen_config)

                if charts:
                    image_base64 = charts[0].raster
                    img = base64_to_image(image_base64)
                    st.image(img)

                    # Update history with current query and image
                    update_history(text_area, img)

    show_history()


_ = """
import streamlit as st
from lida import Manager, TextGenerationConfig, llm
from dotenv import load_dotenv
import os
import openai
from PIL import Image
from io import BytesIO
import base64

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


def base64_to_image(base64_string):
    byte_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(byte_data))


lida = Manager(text_gen=llm("openai"))
textgen_config = TextGenerationConfig(
    n=1, temperature=0.5, model="gpt-3.5-turbo-0301", use_cache=True)

st.sidebar.title("Visualization ChatBot")

menu = st.sidebar.selectbox(
    "Choose an Option", ["Generate Graph", "Summarize"])

st.subheader("Query your Data to Generate Graph")
file_uploader = st.file_uploader("Upload your CSV", type="csv")

if file_uploader is not None:
    path_to_save = "filename1.csv"
    with open(path_to_save, "wb") as f:
        f.write(file_uploader.getvalue())

    text_area = st.text_area(
        "Query your Data to Generate Graph", height=200)

    if st.button("Generate Graph"):
        if len(text_area) > 0:
            st.info("Your Query: " + text_area)

            summary = lida.summarize(
                "filename1.csv", summary_method="default", textgen_config=textgen_config)
            user_query = text_area
            charts = lida.visualize(
                summary=summary, goal=user_query, textgen_config=textgen_config)

            if charts:
                image_base64 = charts[0].raster
                img = base64_to_image(image_base64)
                st.image(img)
"""
