import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import tempfile
import os

# Streamlit UI setup
st.set_page_config(page_title='Akash PDF Summarizer')
st.title("📄 Akash - PDF Summarizer")
st.subheader("Summarize any PDF using Groq's LLaMA 3 🚀")
st.divider()

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Summarize Button
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Reading and summarizing PDF..."):
        try:
            # Load PDF and extract text
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            # Combine all pages into one string
            full_text = "\n".join([page.page_content for page in pages])

            # Split text
            text_splitter = CharacterTextSplitter()
            texts = text_splitter.split_text(full_text)
            docs = [Document(page_content=t) for t in texts]

            # Groq LLM via LangChain
            llm = ChatGroq(
                model_name="llama3-8b-8192",
                temperature=0,
                groq_api_key=st.secrets["GROQ_API_KEY"]
            )

            # Summarization Chain
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            summary = chain.run(docs)

            # Display result
            st.success("✅ Summary Generated:")
            st.write(summary)

        except Exception as e:
            st.error(f"❌ Error: {e}")

        finally:
            os.remove(tmp_path)  # Clean up temp file
