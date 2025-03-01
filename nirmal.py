import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configure Google Generative AI
genai.configure(api_key="AIzaSyBsq5Kd5nJgx2fejR77NT8v5Lk3PK4gbH8")  
gemini = genai.GenerativeModel('gemini-1.5-flash')

# Load the embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2') 

# Load the dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('my_data.csv')  # Replace with your dataset file name
        if 'question' not in df.columns or 'answer' not in df.columns:
            st.error("The CSV file must contain 'question' and 'answer' columns.")
            st.stop()
        # Create embeddings for all questions in the dataset
        df['embedding'] = list(embedder.encode(df['question'].tolist()))
        return df
    except Exception as e:
        st.error(f"Failed to load data. Error: {e}")
        st.stop()

df = load_data()

# Streamlit UI
st.markdown('<h1>🤖 Nirmal Gaud Clone Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<h3>Ask me anything, and I\'ll respond as Nirmal Gaud!</h3>', unsafe_allow_html=True)
st.markdown("---")

# Function to find the best match
def find_best_match(query, df, similarity_threshold=0.7):
    # Encode the query
    query_embedding = embedder.encode([query])
    
    # Compute cosine similarity between the query and all questions in the dataset
    similarities = cosine_similarity(query_embedding, np.stack(df['embedding']))
    max_similarity_index = np.argmax(similarities)
    max_similarity = similarities[0][max_similarity_index]
    
    # Check if the similarity meets the threshold
    if max_similarity >= similarity_threshold:
        return df.iloc[max_similarity_index]['answer'], max_similarity
    return None, 0

# Function to refine the answer using Gemini
def refine_answer_with_gemini(query, retrieved_answer):
    prompt = f"""You are Nirmal Gaud, an AI, ML, and DL instructor. Respond to the following question in a friendly and conversational tone:
    Question: {query}
    Retrieved Answer: {retrieved_answer}
    - Do not add any new information.
    - Ensure the response is grammatically correct and engaging.
    """
    response = gemini.generate_content(prompt)
    return response.text

# Chatbot logic
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
            # Find the best match
            retrieved_answer, similarity_score = find_best_match(prompt, df, similarity_threshold=0.7)
            if retrieved_answer:
                # Refine the answer using Gemini
                refined_answer = refine_answer_with_gemini(prompt, retrieved_answer)
                response = f"**Nirmal Gaud**:\n{refined_answer}"
            else:
                response = "**Nirmal Gaud**:\nThis is out of context. Please ask something related to my dataset."
        except Exception as e:
            response = f"An error occurred: {e}"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
