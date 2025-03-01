import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- Styling ---
st.markdown("""
<style>
    .stApp {
        background: #f8f5e6;
        background-image: radial-gradient(#d4d0c4 1px, transparent 1px);
        background-size: 20px 20px;
    }
    .chat-font {
        font-family: 'Times New Roman', serif;
        color: #2c5f2d;
    }
    .user-msg {
        background: #ffffff !important;
        border-radius: 15px !important;
        border: 2px solid #2c5f2d !important;
        padding: 10px;
    }
    .bot-msg {
        background: #fff9e6 !important;
        border-radius: 15px !important;
        border: 2px solid #ffd700 !important;
        padding: 10px;
    }
    .stChatInput {
        background: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# --- API Keys and Models ---
genai.configure(api_key="YOUR_API_KEY")  # Replace with your actual API key
gemini = genai.GenerativeModel('gemini-1.5-flash')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# --- Data Loading and Indexing ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('my_data.csv')
        if 'question' not in df.columns or 'answer' not in df.columns:
            st.error("The CSV file must contain 'question' and 'answer' columns.")
            st.stop()
        df['context'] = df.apply(
            lambda row: f"Question: {row['question']}\nAnswer: {row['answer']}",
            axis=1
        )
        embeddings = embedder.encode(df['context'].tolist())
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings).astype('float32'))
        return df, index
    except FileNotFoundError:
        st.error("CSV file 'my_data.csv' not found. Please ensure it exists in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load data. Error: {e}")
        st.stop()

df, faiss_index = load_data()

# --- UI Setup ---
st.markdown('<h1 class="chat-font"> Nirmal Gaud Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="chat-font">Ask me anything, and I\'ll respond as myself, Nirmal Gaud.</h3>', unsafe_allow_html=True)
st.markdown("---")

# --- Helper Functions ---
def find_closest_question(query, faiss_index, df):
    query_embedding = embedder.encode([query])
    distances, indices = faiss_index.search(query_embedding.astype('float32'), k=1)
    if indices.size > 0:
        return df.iloc[indices[0][0]]['answer']
    return None

def generate_refined_answer(query, retrieved_answer):
    prompt = f"""You are Nirmal Gaud, a human AI, ML, and DL instructor. Respond to the following question in a friendly and conversational tone, maintaining your personal style and perspective.
Question: {query}
Retrieved Answer: {retrieved_answer}
- Provide a detailed and accurate response, as you would naturally.
- Ensure the response is grammatically correct and engaging.
- Express your own opinions and experiences where relevant.
"""
    try:
        response = gemini.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.2))
        return response.text
    except Exception as e:
        return f"Error generating response: {e}"

# --- Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=None if message["role"] == "user" else None):
        st.markdown(message["content"], unsafe_allow_html=True)

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        try:
            retrieved_answer = find_closest_question(prompt, faiss_index, df)
            if retrieved_answer:
                refined_answer = generate_refined_answer(prompt, retrieved_answer)
                response = f"**Nirmal Gaud**:\n{refined_answer}"
            else:
                response = "**Nirmal Gaud**:\nI'm sorry, I don't have a specific answer for that."
        except Exception as e:
            response = f"An error occurred: {e}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
