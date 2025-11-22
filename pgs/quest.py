import streamlit as st 
import sys
import tempfile
import os



sys.path.insert(1, './modules')

from upload_file_rag import get_qa_chain, query_system






# st.image('WhatsApp Image 2025-08-11 at 10.12.18 AM.jpeg', width=900)
st.image('https://cdn.imweb.me/upload/S202509027fa77ca22ea24/9158a17980c28.png', width=900)

# col1, col2 = st.columns(2)

# with col1:
def reset_conversation():
  st.session_state.conversation = None
  st.session_state.chat_history = None


with st.sidebar:
    if st.button(label="", icon=":material/quick_reference_all:", on_click=reset_conversation):
        with st.spinner("Refreshing chat... Please wait."):
            st.success("Chat refreshed successfully!")


with st.sidebar:
    uploaded_files = st.file_uploader('Upload Document', accept_multiple_files=True, type='pdf')

if uploaded_files:
    for uploaded_file in uploaded_files:

        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        # Initialize QA chain from saved file
        qa_chain = get_qa_chain(temp_path)

        # Initialize session state for chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

        # Display chat history
        for message in st.session_state.messages:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])


        if prompt := st.chat_input("How may I help?", key='RAG chat'):
            # Append user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate AI response
            chat_output = query_system(prompt, qa_chain)
            
            # Append AI response
            with st.chat_message("assistant"):
                st.markdown(chat_output)

            st.session_state.messages.append({"role": "assistant", "content": chat_output})



