import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.core.settings import Settings

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# Initialize the LLM with increased timeout
llm = Ollama(model="llama3", request_timeout=60)  # adjust timeout if needed

# Set global settings
Settings.llm = llm
# Optional: Add embed_model if you're using one
# Settings.embed_model = YourEmbedModel()

# Build index
index = VectorStoreIndex.from_documents(documents)
chat_engine = index.as_chat_engine(chat_mode="react", verbose=True)

# Streamlit interface
st.set_page_config(page_title="Real Estate Chatbot", layout="centered")
st.title("🏡 Real Estate Chatbot")

# Get user input
user_input = st.text_input("Ask something about the real estate data:")

# Generate and display response
if user_input:
    try:
        response = chat_engine.chat(user_input)
        st.markdown(f"**Bot:** {response.response}")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
