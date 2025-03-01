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
genai.configure(api_key="AIzaSyBsq5Kd5nJgx2fejR77NT8v5Lk3PK4gbH8")  # Replace with your actual API key
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
        # Create embeddings for all questions in the dataset
        embeddings = embedder.encode(df['question'].tolist())
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Build FAISS index
        index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product for cosine similarity
        index.add(embeddings)
        return df, index
    except FileNotFoundError:
        st.error("CSV file 'my_data.csv' not found. Please ensure it exists in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load data. Error: {e}")
        st.stop()

df, faiss_index = load_data()

# --- UI Setup ---
st.markdown('<h1 class="chat-font">🤖 Nirmal Gaud Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="chat-font">Ask me anything, and I\'ll respond as myself, Nirmal Gaud.</h3>', unsafe_allow_html=True)
st.markdown("---")

# --- Helper Functions ---
def find_closest_question(query, faiss_index, df, similarity_threshold=0.7):
    # Encode the query
    query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    
    # Normalize the query embedding
    faiss.normalize_L2(query_embedding)
    
    # Search for the closest match using FAISS
    D, I = faiss_index.search(query_embedding, k=1)  # Top 1 match
    if I.size > 0:
        max_similarity = D[0][0]  # Cosine similarity score
        if max_similarity >= similarity_threshold:
            return df.iloc[I[0][0]]['answer'], max_similarity
    return None, 0

def generate_refined_answer(query, retrieved_answer):
    # Use Gemini to make the response more conversational
    prompt = f"""You are Nirmal Gaud, an AI, ML, and DL instructor. Respond to the following question in a friendly and conversational tone:
    Question: {query}
    Retrieved Answer: {retrieved_answer}
    - Do not add any new information.
    - Ensure the response is grammatically correct and engaging.
    """
    response = gemini.generate_content(prompt)
    return response.text

# --- Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], 
                        avatar="🙋" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        try:
            # Find the closest answer
            retrieved_answer, similarity_score = find_closest_question(prompt, faiss_index, df, similarity_threshold=0.7)
            if retrieved_answer:
                # Generate a refined answer using Gemini
                refined_answer = generate_refined_answer(prompt, retrieved_answer)
                response = f"**Nirmal Gaud**:\n{refined_answer}"
            else:
                response = "**Nirmal Gaud**:\nThis is out of context. Please ask something related to my dataset."
        except Exception as e:
            response = f"An error occurred: {e}"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
