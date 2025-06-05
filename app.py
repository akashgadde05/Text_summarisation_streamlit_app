import streamlit as st
from groq import Groq
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Initialize Groq client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def generate_chunked_summary(text):
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0, groq_api_key=st.secrets["GROQ_API_KEY"])

    # Split text into smaller chunks (~2000 tokens)
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = splitter.split_text(text)

    docs = [Document(page_content=t) for t in texts]

    # Use map_reduce summarization chain for chunked summary
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)

    return summary

st.title("PDF Text Summarization")

# Text input (replace this with your PDF text extractor)
pdf_text = st.text_area("Paste extracted PDF text here:", height=300)

if st.button("Summarize"):
    if not pdf_text.strip():
        st.warning("Please enter some text to summarize!")
    else:
        with st.spinner("Generating summary..."):
            summary = generate_chunked_summary(pdf_text)
        st.success("Summary generated!")
        st.write(summary)
