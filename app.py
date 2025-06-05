import streamlit as st
from groq import Groq
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from fpdf import FPDF  # PDF generation

# Page setup
st.set_page_config(page_title='AKASH - Text Summarization App')

# Display logo
st.image("IMG20241026120028[1].jpg")
st.divider()
st.title('AKASH - Text Summarization App')
st.divider()

# Initialize Groq client with API key stored in Streamlit secrets
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def generate_response(text: str) -> str:
    """Use Groq-powered LLM to summarize text."""
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0,
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )
    # Split the input text into chunks
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    # Create summarization chain and run
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)

def create_pdf(text: str) -> bytes:
    """Generate a PDF file from text and return bytes."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # Split long text into lines of max 90 chars for PDF
    lines = text.split('\n')
    for line in lines:
        pdf.multi_cell(0, 10, line)
    return pdf.output(dest='S').encode('latin1')  # Return PDF as bytes

# Text input area
txt_input = st.text_area('Enter your text here for summarization:', height=200)

result = []

# Form for submission
if txt_input:
    with st.form('summarize_form', clear_on_submit=True):
        submitted = st.form_submit_button('Summarize')
        if submitted:
            with st.spinner('Generating summary...'):
                summary = generate_response(txt_input)
                result.append(summary)

# Display the summary and download button
if result:
    st.info(result[0])

    pdf_bytes = create_pdf(result[0])
    st.download_button(
        label="Download Summary as PDF",
        data=pdf_bytes,
        file_name="summary.pdf",
        mime="application/pdf"
    )
