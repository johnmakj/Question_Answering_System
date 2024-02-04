from typing import Tuple, Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.machine_learning.training.bert_qa_finetuning import BertQAFinetuning
from src.machine_learning.training.bert_qa_layer import CustomQuestionAnsweringModel
from src.machine_learning.training.document_retrieval import DocumentRetrieval
from src.tools.general_tools import load_pickled_data, get_filepath, load_yaml_config

DEFAULT_CONFIG_PREPROCESSING_PATH = get_filepath('demo', "demo_config.yaml")

config = load_yaml_config(DEFAULT_CONFIG_PREPROCESSING_PATH)

data = load_pickled_data(get_filepath("results/data_preprocessing", config["demo"]["dataset"]))
nlp = load_pickled_data(get_filepath("results/models", config["demo"]["nlp"]))
ranker = load_pickled_data(get_filepath("results/models", config["demo"]["ranker"]))
model = CustomQuestionAnsweringModel.from_pretrained(config["demo"]["bert_file"])


def question_answering(question: str):
    relevant_document = DocumentRetrieval(data, nlp).retrieve_document(question, ranker)
    answer = BertQAFinetuning.question_answering(model, question, relevant_document)
    return relevant_document, answer


def ChangeWidgetFontSize(wgt_txt, wch_font_size='12px'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                    for (i = 0; i < elements.length; ++i) { if (elements[i].innerText == |wgt_txt|) 
                        { elements[i].style.fontSize='""" + wch_font_size + """';} } </script>  """

    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)


# App
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

col1_im, col2_im, col3_im = st.columns([0.3, 0.6, 0.3], gap="large")

with col2_im:
    st.title('Question Answering System')

st.markdown(
    """
    <style>
        [data-testid=stSidebar] {
            background-color: #001F3F;
        }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    label_color = "white"

    label_style = f"""
        <style>
            .stTextInput label {{
                color: {label_color};
            }}
        </style>
    """
    st.markdown(label_style, unsafe_allow_html=True)
    query = st.text_input(label="Ask me a Question!")
    ChangeWidgetFontSize('Query', '24px')
    submit = st.button("Answer")

if submit:
    if query == '':
        st.markdown(f"#### You need ask a question first ")
    else:
        rel_doc = st.markdown(f"#### Most relevant document: ")
        relevant_doc, answer = question_answering(query)
        st.markdown(relevant_doc, unsafe_allow_html=True)
        final_answer = st.markdown(f"#### Answer: ")
        if len(answer):
            st.markdown(answer, unsafe_allow_html=True)
        else:
            st.markdown("Cannot provide answer to your question", unsafe_allow_html=True)
if query:
    st.write("")
