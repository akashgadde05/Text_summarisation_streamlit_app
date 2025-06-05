import streamlit as st
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq

st.set_page_config(page_title='PDF Text Summarization App')

# Your Groq API Key from secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    pdf = PdfReader(pdf_file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

def generate_response(txt):
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0, groq_api_key=GROQ_API_KEY)
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    docs = [Document(page_content=t) for t in texts]
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)

st.title("PDF Text Summarization App")

uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_pdf is not None:
    with st.spinner("Extracting text from PDF..."):
        extracted_text = extract_text_from_pdf(uploaded_pdf)

    if extracted_text.strip():
        st.subheader("Extracted Text")
        st.write(extracted_text[:1000] + ("..." if len(extracted_text) > 1000 else ""))  # preview first 1000 chars

        if st.button("Summarize Extracted Text"):
            with st.spinner("Summarizing..."):
                summary = generate_response(extracted_text)
            st.subheader("Summary")
            st.write(summary)

            # Download summary as text file
            st.download_button(
                label="Download Summary as TXT",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )
    else:
        st.error("No text could be extracted from the uploaded PDF.")
else:
    st.info("Please upload a PDF file to get started.")
