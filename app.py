import streamlit as st
import requests
import re
import io
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os
# API configuration
url = "https://proxy.tune.app/chat/completions"
headers = {
    "Authorization": "sk-tune-V05AcvTVpDt7GrJj4Th23QyBb95alb4XVfT",
    "Content-Type": "application/json",
}
background_color = "#1E1E1E"  # Dark gray, almost black
text_color = "#E0E0E0"  # Light gray for text
primary_color = "#4CAF50"  # Chrome green
secondary_color = "#388E3C"  # Darker green for hover effects
accent_color = "#8BC34A"  # Lighter green for accents
input_bg_color = "#2C2C2C"  # Slightly lighter than background for input fields
info_bg_color = "#2C3E50"  # Dark blue-gray for info box
info_text_color = "#FFFFFF"  # Very light blue-gray for info text
font = "Roboto, sans-serif"  # Modern, clean font

# Custom CSS to style the app
custom_css = """
<style>
    /* Modern font import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #FFFFFF;
        color: #333333;
    }

    /* Header styles */
    h1, h2, h3 {
        color: #000000;
        font-weight: 600;
    }

    /* Button styles */
    .stButton > button {
        background-color: #007AFF;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover {
        background-color: #0056b3;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }

    /* Input field styles */
    .stTextInput > div > div > input {
        background-color: #F5F5F7;
        color: #333333;
        border: 1px solid #D2D2D7;
        border-radius: 8px;
        padding: 10px 15px;
    }

    /* Info box styles */
    .stInfo {
        background-color: #F5F5F7 !important;
        color: #1D1D1F !important;
        border: none;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    /* Text color */
    p, li {
        color: #333333 !important;
    }

    /* Spinner color */
    .stSpinner > div {
        border-top-color: #007AFF !important;
    }

    /* Streamlit native elements */
    .css-145kmo2 {
        border-color: #D2D2D7 !important;
    }
    .css-1kyxreq {
        color: #007AFF !important;
    }

    /* Custom gradient background for header */
    .header-gradient {
        background: linear-gradient(90deg, #007AFF, #00C7BE);
        color: white;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
</style>
"""

def set_page_config():
    st.set_page_config(
        page_title="LLama Concept Explorer",
        page_icon="🦙",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)


def make_api_call(messages, stream=False, max_tokens=2000):
    data = {
        "temperature": 0.7,
        "messages": messages,
        "model": "meta/llama-3.2-90b-vision",
        "stream": stream,
        "frequency_penalty": 0.2,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(url, headers=headers, json=data)

        # Check for errors
        if response.status_code != 200:
            st.error(f"API call failed with status code: {response.status_code}")
            st.error(f"Response text: {response.text}")
            return None

        # Attempt to decode the JSON response
        response_json = response.json()

        if stream:
            return response.iter_lines()

        assistant_response = response_json['choices'][0]['message']['content']

        # Check if response is cut off and continue if necessary
        while response_json['choices'][0]['finish_reason'] != 'stop':
            continuation_messages = messages + [{"role": "assistant", "content": assistant_response}]
            continuation_data = {
                "temperature": 0.7,
                "messages": continuation_messages,
                "model": "meta/llama-3.2-90b-vision",
                "stream": stream,
                "frequency_penalty": 0.2,
                "max_tokens": max_tokens
            }
            continuation_response = requests.post(url, headers=headers, json=continuation_data)
            continuation_json = continuation_response.json()
            assistant_response += continuation_json['choices'][0]['message']['content']
            response_json = continuation_json

        return {"choices": [{"message": {"content": assistant_response}}]}

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        return None

    except ValueError:
        st.error("Failed to decode the response as JSON")
        st.error(f"Response content: {response.text}")
        return None


def generate_questions(user_query):
    messages = [
        {"role": "system", "content": "You are an AI Prompt Engineer"},
        {"role": "user", "content": f"""Generate 5 fill-in-the-blank questions to help refine and clarify the user's query: '{user_query}'.
        These questions should aim to gather more specific information about the user's needs, preferences, or context.
        
        Each question should focus on a distinct aspect of the user's query that could help in providing a more tailored response.
        Ensure the questions range from general to specific to gauge the depth of the user's requirements.
        
        Format each question as follows:
        1. [Question]
        
        2. [Next Question]...
        
        Provide only the questions. Don't include answers or any additional text. Try to complete it in the same line."""}
    ]
    response = make_api_call(messages, max_tokens=200)
    questions = response['choices'][0]['message']['content'].strip().split('\n\n')
    return [q.strip() for q in questions if q.strip() and q[0].isdigit()]

def create_quiz(questions):
    user_answers = []
    for i, question in enumerate(questions):
        q_text = question.split('.', 1)[1].strip()
        
        # Use columns to create a more compact layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Display question with smaller font and remove bottom margin
            st.markdown(f"<small>Q{i+1}: {q_text}</small>", unsafe_allow_html=True)
        
        with col2:
            # Add a small vertical space to align with the question
            st.write("")
        
        # Place the text input directly below the question
        key = f"answer_{i}"
        if key not in st.session_state:
            st.session_state[key] = ""
        
        answer = st.text_input("", key=key, label_visibility="collapsed")
        user_answers.append(answer)
        
        # Add a small vertical space between questions
        st.write("")
    
    return user_answers

def analyze_understanding(original_prompt, questions, user_answers):
    refinement_prompt = f"""Given the original user query: '{original_prompt}'
    and the following fill-in-the-blank questions and answers:

    """
    
    for q, ua in zip(questions, user_answers):
        q_text = q.split('.', 1)[1].strip()
        refinement_prompt += f"Q: {q_text}\nA: {ua}\n\n"
    
    refinement_prompt += """Based on this information, generate a refined and more specific version of the original query.
    The refined prompt should:
    1. Incorporate the additional context and preferences expressed in the answers.
    2. Be more detailed and targeted than the original query.
    3. Address any ambiguities or uncertainties resolved by the answers.
    4. Be phrased as a clear, concise request or question.
    
    Provide only the refined prompt, without any additional explanation or context."""

    messages = [
        {"role": "system", "content": "You are an AI Prompt Engineer"},
        {"role": "user", "content": refinement_prompt}
    ]
    
    response = make_api_call(messages)
    refined_prompt = response['choices'][0]['message']['content'].strip()
    
    return refined_prompt

def create_persona(user_query, questions, user_answers, refined_prompt):
    persona = io.StringIO()
    persona.write(f"Original Query: {user_query}\n\n")
    persona.write("User Responses:\n")
    for q, a in zip(questions, user_answers):
        q_text = q.split('.', 1)[1].strip()
        persona.write(f"Q: {q_text}\n")
        persona.write(f"A: {a}\n\n")
    persona.write(f"Refined Prompt: {refined_prompt}\n")
    return persona.getvalue()

import requests
from typing import List,Optional
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
import logging
TUNE_API_KEY = "sk-tune-V05AcvTVpDt7GrJj4Th23QyBb95alb4XVfT"
TUNE_API_URL = "https://proxy.tune.app/chat/completions"
class ProxyEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        """Initialize ProxyEmbeddings with an API key."""
        self.api_key = api_key  # Store the passed API key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.api_base = "https://proxy.tune.app/v1"  # API endpoint

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Make an API call to retrieve embeddings for the provided texts."""
        data = {
            "input": texts,
            "model": "openai/text-embedding-ada-002",
            "encoding_format": "float"
        }
        response = requests.post(f"{self.api_base}/embeddings", headers=self.headers, json=data)
        response.raise_for_status()  # Raises an error for any bad response
        return [d['embedding'] for d in response.json()['data']]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (texts)."""
        if not texts:
            return []
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        if not text:
            return [0.0] * 1536  # Assuming the embedding length is 1536
        return self._get_embeddings([text])[0]

    def __repr__(self):
        return f"ProxyEmbeddings(api_base='{self.api_base}')"
class ProxyLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            response = requests.post(
                TUNE_API_URL,
                headers={
                    "Authorization": TUNE_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "model": "meta/llama-3.2-90b-vision",
                    "temperature": 0.7,
                    "max_tokens": 500
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            st.error(f"LLM API error: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again in a moment."

    @property
    def _llm_type(self) -> str:
        return "proxy_llm"
# Initialize embeddings
#embeddings = ProxyEmbeddings(api_key="sk-tune-V05AcvTVpDt7GrJj4Th23QyBb95alb4XVfT")

def setup_rag(persona: str) -> Optional[RetrievalQA]:
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Increase chunk size and overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Increased from 100
            chunk_overlap=50,  # Increased from 20
            length_function=len
        )
        texts = text_splitter.create_documents([persona])
        logger.info(f"Created {len(texts)} text chunks")

        # Set up embeddings
        try:
            embeddings = ProxyEmbeddings(api_key="sk-tune-V05AcvTVpDt7GrJj4Th23QyBb95alb4XVfT")
            logger.info("Embeddings created successfully")
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise

        # Create vector store
        try:
            db = FAISS.from_documents(texts, embeddings)
            retriever = db.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 chunks
            logger.info("Vector store created successfully")
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise

        # Set up LLM
        try:
            llm = ProxyLLM()
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        logger.info("QA chain created successfully")

        return qa_chain
    except Exception as e:
        logger.error(f"Error setting up RAG system: {str(e)}")
        st.error(f"An error occurred while setting up the system. Please try again or contact support if the issue persists.")
        return None

def generate_tailored_explanation(qa_chain: RetrievalQA, refined_prompt: str) -> str:
    try:
        # Log the refined prompt
        logging.info(f"Generating explanation for refined prompt: {refined_prompt}")

        # Use the RAG system to generate the explanation
        rag_response = qa_chain({"query": refined_prompt})
        explanation = rag_response['result']

        # Log the generated explanation
        logging.info(f"Generated explanation: {explanation[:100]}...")  # Log first 100 chars

        if not explanation or explanation.lower().startswith("i don't know"):
            # If the explanation is empty or starts with "I don't know", generate a fallback response
            fallback_prompt = refined_prompt

            fallback_response = qa_chain({"query": fallback_prompt})
            explanation = fallback_response['result']
            logging.info("Used fallback method to generate explanation")

        return explanation

    except Exception as e:
        logging.error(f"Error generating tailored explanation: {str(e)}")
        return "I apologize, but I'm having trouble generating a detailed explanation right now. Please try rephrasing your query or try again later."

def llama_tab():
    st.subheader("LLama Concept Explorer and Tailored Explanation")

    # Initialize session state variables
    if 'knowledge_mapped' not in st.session_state:
        st.session_state.knowledge_mapped = False
    if 'user_query' not in st.session_state:
        st.session_state.user_query = ""
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'refined_prompt' not in st.session_state:
        st.session_state.refined_prompt = ""
    if 'explanation' not in st.session_state:
        st.session_state.explanation = ""
    if 'show_refined_prompt' not in st.session_state:
        st.session_state.show_refined_prompt = False

    user_query = st.text_input("Enter a concept or topic you want to explore:",
                                key="user_query_input",
                                value=st.session_state.user_query)

    if user_query != st.session_state.user_query:
        st.session_state.user_query = user_query
        st.session_state.knowledge_mapped = False
        st.session_state.questions = []
        st.session_state.refined_prompt = ""
        st.session_state.explanation = ""
        st.session_state.show_refined_prompt = False

    if st.session_state.user_query and not st.session_state.knowledge_mapped:
        if st.button("Map Your Knowledge", key="map_knowledge_button"):
            with st.spinner("Crafting your knowledge map..."):
                st.session_state.questions = generate_questions(st.session_state.user_query)
                st.session_state.knowledge_mapped = True

    if st.session_state.knowledge_mapped:
        user_answers = create_quiz(st.session_state.questions)
        
        if st.button("Analyze Understanding and Explain", key="analyze_button"):
            if any(not answer.strip() for answer in user_answers):
                st.warning("Please answer all questions to complete your knowledge map.")
            else:
                with st.spinner("Analyzing your understanding and generating personalized insights..."):
                    st.session_state.refined_prompt = analyze_understanding(st.session_state.user_query, st.session_state.questions, user_answers)
                    st.session_state.user_persona = create_persona(st.session_state.user_query, st.session_state.questions, user_answers, st.session_state.refined_prompt)
                    qa_chain = setup_rag(st.session_state.user_persona)
                    st.session_state.explanation = generate_tailored_explanation(qa_chain, st.session_state.refined_prompt)

    if st.session_state.explanation:
        if st.button("Reveal Your Knowledge Profile", key="reveal_button"):
            st.session_state.show_refined_prompt = not st.session_state.show_refined_prompt

        if st.session_state.show_refined_prompt:
            st.info(st.session_state.refined_prompt)

        st.write("Your Personalized Concept Breakdown:")
        st.write(st.session_state.explanation)

    if st.button("Reset Knowledge Map", key="reset_map_button"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

def set_page_config():
    st.set_page_config(
        page_title="LLama Concept Explorer",
        page_icon="🦙",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)

def main():
    set_page_config()
    
    # Add gradient header
    st.markdown("""
        <div class="header-gradient">
            <h1 style="font-size: 36px; font-weight: 600;">Unlock 10x Faster, Smarter and Personalised Answers</h1>
            <p style="font-size: 18px;">Explore concepts with our advanced AI-powered chatbot</p>
        </div>
    """, unsafe_allow_html=True)
    
    llama_tab()

# ... (rest of the code remains the same)

if __name__ == "__main__":
    main()
