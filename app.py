import streamlit as st
import requests
import json
import os
import re

# API configuration
url = "https://proxy.tune.app/chat/completions"
headers = {
    "Authorization": "sk-tune-V05AcvTVpDt7GrJj4Th23QyBb95alb4XVfT",
    "Content-Type": "application/json",
}
def make_api_call(messages, stream=False, max_tokens=300):
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

def extract_keyword(prompt):
    messages = [
        {"role": "system", "content": "You are TuneStudio"},
        {"role": "user", "content": f"Extract the main keyword or concept from the following query. Respond with only the keyword or concept, nothing else. Query: {prompt}"}
    ]
    response = make_api_call(messages)
    return response['choices'][0]['message']['content'].strip()

def generate_questions(keyword):
    messages = [
        {"role": "system", "content": "You are TuneStudio"},
        {"role": "user", "content": f"""Generate a mix of 10 questions about key concepts related to '{keyword}' in the context of machine learning or AI. 
        Include the following mix:
        - 4 Yes/No questions
        - 3 Multiple choice questions (with 4 options each)
        - 3 True/False questions
        
        Each question should focus on a distinct sub-concept or aspect of {keyword}.
        Ensure the questions range from basic to advanced concepts to gauge the user's depth of understanding.
        
        Format each question as follows:
        1. [Question] (Type: Y/N, MC, or T/F)
        [For MC questions, list options as:
        a) [option]
        b) [option]
        c) [option]
        d) [option]]
        (Correct Answer)
        
        2. [Next Question]...
        This is how an MC question can look-
        Q)How do Transformers typically handle longer input sequences that exceed their maximum context length?

        a) By using sliding window attention
        b) By truncating the input to fit the maximum length
        c) By using recurrent layers to process the sequence iteratively
        d) By automatically increasing the model's context length
        This is how a T/F question can look-
        True or False: The Transformer architecture relies on convolutional layers.
        a)True
        b)False
        This is how a Y/N question can look-
        Are Transformers typically used for sequence-to-sequence tasks?
        a)Yes
        b)No
        
        Provide only the questions, options (for MC), and correct answers. For a Y/N question, provide yes or no options but if the question is T/F provide True or False options. If the question is MC, provide the options with one of them being the right answer. Don't provide a  No additional text."""}
    ]
    response = make_api_call(messages)
    questions = response['choices'][0]['message']['content'].strip().split('\n\n')
    return [q.strip() for q in questions if q.strip() and q[0].isdigit()]

def create_quiz(questions):
    user_answers = []
    for i, question in enumerate(questions):
        lines = question.split('\n')
        q_text = lines[0].split('.', 1)[1].split('(')[0].strip()
        q_type = lines[0].split('(')[1].split(')')[0].strip()
        
        st.write(f"Q{i+1}: {q_text}")
        
        if q_type == "Y/N":
            options = ["Yes", "No"]
        elif q_type == "T/F":
            options = ["True", "False"]
        elif q_type == "MC":
            options = [re.sub(r'^[a-d]\)\s*', '', line.strip()) for line in lines[1:-1]]
        else:
            st.warning(f"Unknown question type for Q{i+1}")
            continue
        
        key = f"answer_{i}"
        if key not in st.session_state:
            st.session_state[key] = None
        
        answer = st.radio(f"Answer {i+1}", options, key=key, index=None)
        user_answers.append(answer)
    
    return user_answers

def analyze_understanding(keyword, questions, user_answers):
    analysis_prompt = f"Based on the following questions and answers about '{keyword}', identify which concepts the user understands and which they don't. Questions and user answers:\n"
    for q, ua in zip(questions, user_answers):
        lines = q.split('\n')
        q_text = lines[0].split('.', 1)[1].split('(')[0].strip()
        correct_answer = lines[-1].strip()[1:-1]  # Remove parentheses
        analysis_prompt += f"Q: {q_text}\nUser's Answer: {ua}\nCorrect Answer: {correct_answer}\n\n"
    analysis_prompt += "\nList the concepts understood and not understood. Format: Understood: [list], Not Understood: [list]"
    
    messages = [
        {"role": "system", "content": "You are TuneStudio"},
        {"role": "user", "content": analysis_prompt}
    ]
    response = make_api_call(messages)
    return response['choices'][0]['message']['content']


def generate_tailored_explanation(keyword, understood, not_understood):
    explanation_prompt = f"""Provide a tailored explanation of {keyword} for someone who understands {understood} but doesn't understand {not_understood}. 
    Explain {keyword} in depth, focusing on explaining {not_understood} thoroughly. 
    Only briefly mention {understood} without going into detail. 
    Keep the explanation clear and concise, suitable for someone learning about {keyword}."""
    
    messages = [
        {"role": "system", "content": "You are TuneStudio"},
        {"role": "user", "content": explanation_prompt}
    ]
    response = make_api_call(messages)
    return response['choices'][0]['message']['content']

def llama_tab():
    st.subheader("LLama Concept Quiz and Tailored Explanation")

    # Initialize session state variables
    if 'quiz_generated' not in st.session_state:
        st.session_state.quiz_generated = False
    if 'keyword' not in st.session_state:
        st.session_state.keyword = ""
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'analysis' not in st.session_state:
        st.session_state.analysis = ""
    if 'explanation' not in st.session_state:
        st.session_state.explanation = ""

    # User query
    user_query = st.text_input("Enter a concept or topic you want to learn about:", key="query")

    if user_query and not st.session_state.quiz_generated:
        # Extract keyword and generate questions
        if st.button("Generate Quiz"):
            with st.spinner("Extracting main concept and generating questions..."):
                st.session_state.keyword = extract_keyword(user_query)
                st.session_state.questions = generate_questions(st.session_state.keyword)
                st.session_state.quiz_generated = True

    if st.session_state.quiz_generated:
        st.write(f"Main concept identified: {st.session_state.keyword}")
        user_answers = create_quiz(st.session_state.questions)
        
        if st.button("Submit Quiz and Get Explanation"):
            if None in user_answers:
                st.warning("Please answer all questions before submitting.")
            else:
                with st.spinner("Analyzing your understanding and generating explanation..."):
                    st.session_state.analysis = analyze_understanding(st.session_state.keyword, st.session_state.questions, user_answers)
                    analysis_parts = st.session_state.analysis.split("Not Understood:")
                    if len(analysis_parts) == 2:
                        understood = analysis_parts[0].split("Understood:")[1].strip()
                        not_understood = analysis_parts[1].strip()
                    else:
                        understood = "some concepts"
                        not_understood = "other concepts"
                    st.session_state.explanation = generate_tailored_explanation(st.session_state.keyword, understood, not_understood)

        if st.session_state.explanation:
            st.write("Tailored Explanation:")
            st.write(st.session_state.explanation)

        if st.button("Reset Quiz"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.quiz_generated = False
            st.session_state.explanation = ""

# Update your main function to use this new llama_tab function
def main():
    st.header("LLM Models", divider="rainbow")
    
    tab0, tab1 = st.tabs(["LLama Concept Quiz", "Generate story"])
    
    with tab0:
        llama_tab()
    
    with tab1:
        # Your existing story generation code here
        st.subheader("Generate a story")
        # Add your story generation code here

if __name__ == "__main__":
    main()
