import streamlit as st
import requests
import re
import io
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
import os
# API configuration
url = "https://proxy.tune.app/chat/completions"
headers = {
    "Authorization": "sk-tune-V05AcvTVpDt7GrJj4Th23QyBb95alb4XVfT",
    "Content-Type": "application/json",
}
primary_color = "#4A90E2"  # A nice shade of blue
background_color = "#F0F4F8"  # Light grayish blue
secondary_background_color = "#E1E8ED"  # Slightly darker grayish blue
text_color = "#333333"  # Dark gray for text
font = "sans-serif"

# Custom CSS to style the app
custom_css = f"""
<style>
    .stApp {{
        background-color: {background_color};
        color: {text_color};
        font-family: {font};
    }}
    .stButton>button {{
        background-color: {primary_color};
        color: white;
    }}
    .stTextInput>div>div>input {{
        background-color: {secondary_background_color};
    }}
    .stMarkdown {{
        color: {text_color};
    }}
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
        
        Provide only the questions. Don't include answers or any additional text."""}
    ]
    response = make_api_call(messages)
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

def generate_tailored_explanation(refined_prompt):
    explanation_prompt = f"""Provide a tailored explanation based on the following refined prompt: '{refined_prompt}'.
    The explanation should:
    1. Address the specific aspects mentioned in the refined prompt.
    2. Be clear and concise, suitable for someone learning about the topic.
    3. Focus on the areas that seem to need more clarification based on the refined prompt.
    4. Provide relevant examples or analogies if appropriate.

    Keep the explanation comprehensive yet accessible to someone who has shown interest in learning more about the topic. Don't leave the answers incomplete. You must complete it properly."""
    
    messages = [
        {"role": "system", "content": "You are an AI Educator"},
        {"role": "user", "content": explanation_prompt}
    ]
    response = make_api_call(messages)
    return response['choices'][0]['message']['content']

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

def setup_rag(persona):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len
        )
        texts = text_splitter.create_documents([persona])
        
        embeddings = ProxyEmbeddings(api_key="sk-tune-V05AcvTVpDt7GrJj4Th23QyBb95alb4XVfT")
        db = FAISS.from_documents(texts, embeddings)
        retriever = db.as_retriever()
        
        llm = ProxyLLM()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error setting up RAG system: {str(e)}")
        return None
def llama_tab():
    st.subheader("LLama Concept Explorer and Tailored Explanation")

    # Initialize session state variables
    if 'knowledge_mapped' not in st.session_state:
        st.session_state.knowledge_mapped = False
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'refined_prompt' not in st.session_state:
        st.session_state.refined_prompt = ""
    if 'explanation' not in st.session_state:
        st.session_state.explanation = ""
    if 'user_query' not in st.session_state:
        st.session_state.user_query = ""
    if 'show_refined_prompt' not in st.session_state:
        st.session_state.show_refined_prompt = False
    if 'persona' not in st.session_state:
        st.session_state.persona = ""
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None

    # User query
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
        st.session_state.persona = ""
        st.session_state.qa_chain = None

    if st.session_state.user_query and not st.session_state.knowledge_mapped:
        if st.button("Map Your Knowledge", key="map_knowledge_button"):
            with st.spinner("Crafting your knowledge map..."):
                attempts = 0
                max_attempts = 3
                while attempts < max_attempts:
                    try:
                        st.session_state.questions = generate_questions(st.session_state.user_query)
                        if len(st.session_state.questions) == 5:
                            st.session_state.knowledge_mapped = True
                            break
                        else:
                            raise ValueError("Incorrect number of questions generated")
                    except (IndexError, ValueError) as e:
                        attempts += 1
                        if attempts == max_attempts:
                            st.error(f"Failed to generate a valid knowledge map: {str(e)}. Please try again or rephrase your query.")
                        else:
                            st.warning(f"Attempt {attempts} failed. Trying again...")

    if st.session_state.knowledge_mapped:
        user_answers = create_quiz(st.session_state.questions)
        
        if st.button("Analyze Understanding and Explain", key="analyze_button"):
            if any(not answer.strip() for answer in user_answers):
                st.warning("Please answer all questions to complete your knowledge map.")
            else:
                with st.spinner("Analyzing your understanding and generating personalized insights..."):
                    st.session_state.refined_prompt = analyze_understanding(st.session_state.user_query, st.session_state.questions, user_answers)
                    st.session_state.persona = create_persona(st.session_state.user_query, st.session_state.questions, user_answers, st.session_state.refined_prompt)
                    st.session_state.qa_chain = setup_rag(st.session_state.persona)
                    
                    # Generate explanation using RAG
                    rag_response = st.session_state.qa_chain({"query": st.session_state.refined_prompt})
                    st.session_state.explanation = rag_response['result']

    if st.session_state.explanation:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("Reveal Your Knowledge Profile", key="reveal_button"):
                st.session_state.show_refined_prompt = not st.session_state.show_refined_prompt

        if st.session_state.show_refined_prompt:
            st.info(st.session_state.refined_prompt)

        st.write("Your Personalized Concept Breakdown:")
        
        # Split the explanation into paragraphs
        paragraphs = st.session_state.explanation.split('\n\n')
        
        # Display the first paragraph
        st.write(paragraphs[0])
        
        # Create an expander for the rest of the explanation
        with st.expander("Read more"):
            for paragraph in paragraphs[1:]:
                st.write(paragraph)
                st.write("")  # Add some space between paragraphs

    if st.button("Reset Knowledge Map", key="reset_map_button"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.knowledge_mapped = False
        st.session_state.explanation = ""
        st.session_state.user_query = ""
        st.session_state.show_refined_prompt = False
        st.session_state.persona = ""
        st.session_state.qa_chain = None
def main():
    set_page_config()
    st.header("Get 10x specific answers with our chatbot", divider="rainbow")
    llama_tab()

if __name__ == "__main__":
    main()
