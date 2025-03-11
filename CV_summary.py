from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter

import textwrap

load_dotenv()

import streamlit as st

import tempfile
import os

def process_docx(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    loader = Docx2txtLoader(tmp_file_path)

    #load the doc and split into chunks to fit the context size
    text = loader.load_and_split()
    
    os.remove(tmp_file_path)  # Clean up the temp file
    return text

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    pages = loader.load()

    text = "".join(page.page_content for page in pages).replace('\t', ' ')

    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=50
    )
    
    texts = text_splitter.create_documents([text])

    os.remove(tmp_file_path)  # Clean up the temp file
    return texts


def main():
    st.title("CV Summary Generator: ")
    uploaded_file = st.file_uploader("select CV", type=["docx", "pdf"])

    text=""

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1]

        st.write("File Details: ")
        st.write(f"File Name: {uploaded_file.name}")
        print(uploaded_file.name)
        st.write(f"File Type: {file_extension}")

        if file_extension == "docx":
            text = process_docx(uploaded_file)
        elif file_extension == "pdf":
            text = process_pdf(uploaded_file)
        else:
            st.error("Unsupported file format.")
            return
        
        
        llm=OpenAI(temperature=0) #model object

        prompt_template = """
              You have been given a resume to analyze.
              Write the verbose detail of the following:
              {text}

              Details:
        """
        
        prompt = PromptTemplate.from_template(prompt_template)


        #when refine stage comes up, this prompt is going to be used on the verbose details, the inital prompt has come up with.
        #this will determine the final outcome


        #exiting_answer will be the input from intial prompt.
        #text is original doc
        refine_template = (
            "Your job is to produce a final outcome.\n"
            "We have provided you with existing details: {existing_answer}\n"
            "We want a refined version of the existing detail based on the intial details below\n"
            "===============\n"
            "{text}"
            "===============\n"
            "Given the new context, refine the original summary in the following manner:"
            "Name:\n"
            "Email:\n"
            "Key Skills:\n"
            "Last Company:\n"
            "Experience Summary:\n"

        )

        refine_prompt = PromptTemplate.from_template(refine_template)
        chain = load_summarize_chain(
            llm=llm,
            chain_type="refine",
            question_prompt = prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps = True,
            input_key ="input_documents",
            output_key="output_text",
        )
        
        #the text here contains info from CV in multiple docs
        result = chain.invoke({"input_documents":text}, return_only_outputs=True)

        st.write("Resume Summary:\n")

        st.text_area("Text", result['output_text'], height=400)

if __name__ == "__main__":
    main()

