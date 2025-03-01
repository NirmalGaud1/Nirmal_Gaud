import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

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
    }
    .bot-msg {
        background: #fff9e6 !important;
        border-radius: 15px !important;
        border: 2px solid #ffd700 !important;
    }
    .stChatInput {
        background: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

genai.configure(api_key="AIzaSyBsq5Kd5nJgx2fejR77NT8v5Lk3PK4gbH8")  
gemini = genai.GenerativeModel('gemini-1.5-flash')

embedder = SentenceTransformer('all-MiniLM-L6-v2') 

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('my_data.csv')  # Replace with your dataset file name
        if 'question' not in df.columns or 'answer' not in df.columns:
            st.error("The CSV file must contain 'question' and 'answer' columns.")
            st.stop()
        df['context'] = df.apply(
            lambda row: f"Question: {row['question']}\nAnswer: {row['answer']}", 
            axis=1
        )
        embeddings = embedder.encode(df['context'].tolist())
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Use IndexFlatIP for cosine similarity
        index = faiss.IndexFlatIP(embeddings.shape[1])  # FAISS index for cosine similarity
        index.add(embeddings.astype('float32'))
        return df, index
    except Exception as e:
        st.error(f"Failed to load data. Error: {e}")
        st.stop()

df, faiss_index = load_data()

st.markdown('<h1 class="chat-font">🤖 Nirmal Gaud Clone Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="chat-font">Ask me anything, and I\'ll respond as Nirmal Gaud!</h3>', unsafe_allow_html=True)
st.markdown("---")

def find_closest_question(query, faiss_index, df, similarity_threshold=0.7):
    query_embedding = embedder.encode([query])
    
    # Normalize the query embedding for cosine similarity
    faiss.normalize_L2(query_embedding)
    
    # Search for the closest match using cosine similarity
    D, I = faiss_index.search(query_embedding.astype('float32'), k=1)  # Top 1 match
    if I.size > 0:
        cosine_similarity = D[0][0]  # Cosine similarity score
        if cosine_similarity >= similarity_threshold:
            return df.iloc[I[0][0]]['answer'], cosine_similarity  # Return the closest answer and similarity score
    return None, 0

def generate_refined_answer(query, retrieved_answer):
    # Simply return the retrieved answer without further refinement
    return retrieved_answer

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
            retrieved_answer, cosine_similarity = find_closest_question(prompt, faiss_index, df, similarity_threshold=0.7)
            if retrieved_answer:
                # Generate a refined answer using Gemini
                refined_answer = generate_refined_answer(prompt, retrieved_answer)
                response = f"**Nirmal Gaud**:\n{refined_answer}"
            else:
                response = "**Nirmal Gaud**:\nI'm sorry, I don't have enough information to answer that question."
        except Exception as e:
            response = f"An error occurred: {e}"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
