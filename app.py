import streamlit as st 
from groq import Groq
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Page title and layout
st.set_page_config(page_title='AKASH-Text Summarization App')
st.image("IMG20241026120028[1].jpg")
st.divider()
st.title('AKASH-Text Summarization App')
st.divider()

# Get Groq API key from Streamlit secrets
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Max input length for llama3-8b-8192 in free tier
MAX_WORDS = 1500

def generate_response(txt):
    # Initialize the LLM model
    llm = ChatGroq(
        model_name="llama3-8b-8192", 
        temperature=0, 
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )

    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    texts = text_splitter.split_text(txt)
    docs = [Document(page_content=t) for t in texts]

    # Load summarization chain
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)

# Text input
txt_input = st.text_area('Enter your text', '', height=200)

# Trim long input text automatically with a warning
if txt_input and len(txt_input.split()) > MAX_WORDS:
    st.warning(f"⚠️ Your input exceeds {MAX_WORDS} words. Trimming to fit model limits.")
    txt_input = " ".join(txt_input.split()[:MAX_WORDS])

# Handle submit
result = []
with st.form('summarize_form', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted and txt_input.strip():
        with st.spinner('Summarizing...'):
            try:
                response = generate_response(txt_input)
                result.append(response)
            except Exception as e:
                st.error(f"❌ Error: {e}")

# Show result
if result:
    st.success("✅ Summary:")
    st.info(result[0])
