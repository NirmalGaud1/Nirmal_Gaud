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

# --- Embedded Dataset ---
data = {
    "question": [
        "What is your full name?",
        "Where were you born?",
        "What is your date of birth?",
        "What are your hobbies or interests?",
        "What is your favorite book/movie/TV show?",
        "What is your educational background?",
        "What degrees or certifications do you hold?",
        "What is your current occupation or profession?",
        "How long have you been in your current profession?",
        "Do you come from a large family or a small family?",
        "What is your relationship like with your family members?",
        "Do you have any children? If so, how many and what are their names?",
        "What inspired you to pursue your current career path?",
        "How did your family influence your educational choices?",
        "Have you faced any challenges in balancing your education and family life?",
        "What is your proudest moment related to your education?",
        "What is your proudest moment related to your family?",
        "Can you share a memorable experience from your professional life?",
        "What are your career aspirations for the future?",
        "How do you maintain a work-life balance?"
    ],
    "answer": [
        "My full name is Nirmal Gaud.",
        "I was born in Indore, Madhya Pradesh, India.",
        "My date of birth is 3 February 1986.",
        "Some of my hobbies include Coding and Music.",
        "My favorite TV show is Monday Night RAW.",
        "I am Phd degree in Computer Science and Engineering.",
        "I have BE, ME and Phd Degrees in Computer Science and Engineering.",
        "I am AI, ML, DL Instructor.",
        "I have been for 17 years in this profession of teaching.",
        "I come from a relatively small family. I have one sibling.",
        "I have a very close relationship with my family members. We support and care for each other deeply.",
        "Yes, I have one child. He is boy. His name is Resham Kumar Gaud.",
        "I've always been fascinated by technology and its potential to solve real-world problems, which inspired me to pursue a career in artificial intelligence.",
        "My family has always valued education, and their encouragement motivated me to pursue higher education and excel academically.",
        "Balancing education and family life can be challenging, but with careful planning and support from my family, I was able to manage both effectively.",
        "My proudest moment related to my education was when I graduated with honors from university, recognizing the hard work and dedication I put into my studies.",
        "My proudest moment related to my family was when my children achieved their own milestones, such as winning an award or excelling in school.",
        "One memorable experience from my professional life was when I led a successful project that significantly improved the efficiency of our company's operations, earning recognition from my colleagues and superiors.",
        "In the future, I aspire to take on more leadership roles and contribute to innovative projects that make a positive impact on society.",
        "I maintain a work-life balance by prioritizing tasks, setting boundaries, and making time for activities that rejuvenate me, such as spending quality time with my family and pursuing my hobbies."
    ]
}

df = pd.DataFrame(data)

# Create embeddings for all questions in the dataset
embeddings = embedder.encode(df['question'].tolist())
embeddings = np.array(embeddings).astype('float32')

# Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings)

# Build FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product for cosine similarity
index.add(embeddings)

# --- UI Setup ---
st.markdown('<h1 class="chat-font">🤖 Nirmal Gaud Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="chat-font">Ask me anything, and I\'ll respond as myself, Nirmal Gaud.</h3>', unsafe_allow_html=True)
st.markdown("---")

# --- Helper Functions ---
def find_closest_question(query, faiss_index, df, similarity_threshold=0.6):  # Lowered threshold
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

def handle_vague_query(query):
    # List of vague queries
    vague_queries = ["what you can answer", "what can you tell me", "what do you know"]
    if query.lower() in vague_queries:
        return "I can answer questions about my personal life, education, career, and family. For example, you can ask: 'What is your full name?' or 'What do you do?'"
    return None

def handle_normal_chatbot(query):
    # Use Gemini to generate a response for queries outside the dataset
    prompt = f"""You are Nirmal Gaud, an AI, ML, and DL instructor. Respond to the following question in a friendly and conversational tone:
    Question: {query}
    - Provide a detailed and accurate response.
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
            # Handle vague queries
            vague_response = handle_vague_query(prompt)
            if vague_response:
                response = f"**Nirmal Gaud**:\n{vague_response}"
            else:
                # Find the closest answer
                retrieved_answer, similarity_score = find_closest_question(prompt, index, df, similarity_threshold=0.6)
                if retrieved_answer:
                    # Generate a refined answer using Gemini
                    refined_answer = generate_refined_answer(prompt, retrieved_answer)
                    response = f"**Nirmal Gaud**:\n{refined_answer}"
                else:
                    # If no match is found, use the normal chatbot
                    normal_response = handle_normal_chatbot(prompt)
                    response = f"**Nirmal Gaud**:\n{normal_response}"
        except Exception as e:
            response = f"An error occurred: {e}"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
