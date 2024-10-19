import streamlit as st
import requests
import re

# API configuration
url = "https://proxy.tune.app/chat/completions"
headers = {
    "Authorization": "sk-tune-V05AcvTVpDt7GrJj4Th23QyBb95alb4XVfT",
    "Content-Type": "application/json",
}

def make_api_call(messages, stream=False, max_tokens=600):
    data = {
        "temperature": 0.7,
        "messages": messages,
        "model": "meta/llama-3.2-90b-vision",
        "stream": stream,
        "frequency_penalty": 0.2,
        "max_tokens": max_tokens
    }
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    
    if stream:
        return response.iter_lines()
    
    assistant_response = response_json['choices'][0]['message']['content']
    
    # Check if response is cut off
    if response_json['choices'][0]['finish_reason'] != 'stop':
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
    
    return {"choices": [{"message": {"content": assistant_response}}]}


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

    Keep the explanation comprehensive yet accessible to someone who has shown interest in learning more about the topic."""
    
    messages = [
        {"role": "system", "content": "You are an AI Educator"},
        {"role": "user", "content": explanation_prompt}
    ]
    response = make_api_call(messages)
    return response['choices'][0]['message']['content']
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
                    st.session_state.explanation = generate_tailored_explanation(st.session_state.refined_prompt)

        if st.session_state.explanation:
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                if st.button("Refined Prompt", key="reveal_button"):
                    st.session_state.show_refined_prompt = not st.session_state.show_refined_prompt

            if st.session_state.show_refined_prompt:
                st.info(st.session_state.refined_prompt)

            st.write("Your Personalized Concept Breakdown:")
            
            # Split the explanation into paragraphs
            paragraphs = st.session_state.explanation.split('\n\n')
            
            # Display the first paragraph
            st.write(paragraphs[0])
            
            # Create an expander for the rest of the explanation
            with st.expander("Read about your question "):
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
def main():
    st.header("LLM Models", divider="rainbow")
    llama_tab()

if __name__ == "__main__":
    main()
