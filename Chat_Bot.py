from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


#function to load gemini pro model and get response
model= genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question):
    response= chat.send_message(question, stream=True)
    return response

##initialize stramlit app

st.set_page_config(page_title="Q&A Bot")
st.header("Gemini chat bot application")

#initialize seasionn state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []


input = st.text_input("Input :", key="input")
submit = st.button("Ask")

if submit and input:
    response = get_gemini_response(input)
    st.session_state['chat_history'].append(("you", input))
    st.subheader("The response is -")
    for ans in response:
        # Check if the response has a valid text attribute
        if hasattr(ans, 'text') and ans.text:
            st.write(ans.text)  # Display the text
            st.session_state['chat_history'].append(("Bot", ans.text))
        # Check if the response has parts
        elif hasattr(ans, 'parts') and ans.parts:
            for part in ans.parts:
                st.write(part.text)
                st.session_state['chat_history'].append(("Bot", part.text))
        else:
            # Handle the case where the response doesn't contain valid content
            st.write("Unable to process the response")



st.subheader("The chat history is -")
for role, text in st.session_state['chat_history']:
    st.write(f"{role} : {text}")