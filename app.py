import streamlit as st
from PyPDF2 import PdfReader
from groq import Groq
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from fpdf import FPDF

st.set_page_config(page_title='PDF Text Summarization with Groq')

st.title("AKASH - PDF Summarization App")
st.divider()

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def extract_text_from_pdf(pdf_file) -> str:
    """Extract all text from uploaded PDF."""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def generate_summary(text: str) -> str:
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0,
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(docs)

def create_pdf(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    return pdf.output(dest='S').encode('latin1')

uploaded_file = st.file_uploader("Upload a PDF file for summarization", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)

    if pdf_text.strip() == "":
        st.error("No extractable text found in the PDF.")
    else:
        st.subheader("Extracted Text Preview:")
        st.write(pdf_text[:1000] + "..." if len(pdf_text) > 1000 else pdf_text)

        if st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                summary = generate_summary(pdf_text)
                st.success("Summary generated!")
                st.info(summary)

                pdf_bytes = create_pdf(summary)
                st.download_button(
                    label="Download Summary as PDF",
                    data=pdf_bytes,
                    file_name="pdf_summary.pdf",
                    mime="application/pdf"
                )
