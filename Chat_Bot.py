import streamlit as st
import google.generativeai as genai
from PIL import Image
import fitz  # PyMuPDF
import docx
import tempfile
import os

# Set page configuration
st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖", layout="wide")

# Initialize Gemini
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = None
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None

# Sidebar configuration
with st.sidebar:
    st.title("Gemini Chatbot")

    # API Key Section
    st.subheader("Gemini API Configuration")
    api_key = st.text_input("Enter your Gemini API key", type="password")
    if st.button("Submit API Key"):
        try:
            genai.configure(api_key=api_key)
            st.session_state.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            st.session_state.chat_session = st.session_state.gemini_model.start_chat(history=[])
            st.success("API key configured successfully!")
        except Exception as e:
            st.error(f"Error configuring API: {str(e)}")

    st.link_button("Get Gemini API Key", "https://ai.google.dev/")

    # Navigation
    st.subheader("Navigation")
    app_mode = st.radio(
        "Select Mode",
        ["Chat", "File Q&A", "Chat with Search", "Langchain Quickstart",
         "Langchain PromptTemplate", "Chat with Feedback"],
        index=0
    )

    # Source code link
    st.divider()
    st.subheader("About")
    st.link_button("View Source Code", "https://github.com/your-repo")
    st.link_button("Open in GitHub Codespaces", "https://github.com/your-repo")


# Main app functions
def chat_interface():
    st.title("Gemini Chat")
    st.caption("A conversational AI powered by Google Gemini")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Your message"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            if not st.session_state.gemini_model:
                st.error("Please configure your Gemini API key first")
            else:
                try:
                    response = st.session_state.chat_session.send_message(prompt)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")


def file_qa_interface():
    def ocr_image(image_path):
        try:
            img = Image.open(image_path)
            prompt_text = "Extract Text from image"
            response = st.session_state.gemini_model.generate_content([prompt_text, img])
            return response.text
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None

    def pdf_to_text(file_path):
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return None

    def docx_to_text(file_path):
        try:
            doc = docx.Document(file_path)
            text = '\n'.join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return None

    def ocr_pdf(file_path):
        """Perform OCR on each page of a scanned PDF."""
        doc = fitz.open(file_path)
        text = ""

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()  # Get the image of the page
            image_path = f"page_{page_num}.png"
            pix.save(image_path)  # Save the image temporarily

            # Perform OCR on the saved image
            page_text = ocr_image(image_path)
            text += f"\n\nPage {page_num + 1}:\n" + page_text

            os.remove(image_path)  # Clean up the temporary image file
        return text

    def convert_to_machine_readable(file_path, file_ext):
        """Main function to convert documents to machine-readable format."""
        try:
            if file_ext == ".pdf":
                text = pdf_to_text(file_path)
                if not text or not text.strip():  # If no text is found, assume it's scanned
                    with st.spinner("PDF appears to be scanned or image-based. Attempting OCR..."):
                        ocr_text = ocr_pdf(file_path)
                    if ocr_text and ocr_text.strip():  # If OCR was successful
                        st.success("OCR completed successfully!")
                        return ocr_text
                    else:
                        st.error("OCR failed to extract text from the PDF")
                        return None
                return text
            elif file_ext == ".docx":
                return docx_to_text(file_path)
            elif file_ext in ('.png', '.jpg', '.jpeg'):
                return ocr_image(file_path)
            elif file_ext == ".txt":
                with open(file_path, "r") as f:
                    return f.read()
            else:
                raise ValueError("Unsupported file format.")
        except Exception as e:
            st.error(f"Error converting file: {str(e)}")
            return None

    st.title("File Q&A with Gemini")
    st.caption("Upload a file and ask questions about its content")

    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "pdf", "txt", "docx"])

    if uploaded_file:
        ##Clear previous caht
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.file_messages = []
            st.session_state.last_uploaded_file = uploaded_file.name
        # Create a temporary file with the correct extension
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        try:
            # Process the file
            with st.spinner(f"Processing {uploaded_file.name}..."):
                file_content = convert_to_machine_readable(temp_file_path, file_ext)

                if file_content is None:
                    st.error("Failed to process the file")
                    return

                # Store the content in session state
                st.session_state.file_content = file_content
                st.success("File processed successfully!")

                # Display preview
                with st.expander("File Preview"):
                    if file_ext in ('.png', '.jpg', '.jpeg'):
                        st.image(temp_file_path, width=300)
                    else:
                        st.text(file_content[:1000] + ("..." if len(file_content) > 1000 else ""))

                # Initialize chat for this file
                if "file_messages" not in st.session_state:
                    st.session_state.file_messages = []

                # Display existing messages
                for message in st.session_state.file_messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Chat input for questions
                if prompt := st.chat_input("Ask about the file..."):
                    st.session_state.file_messages.append({"role": "user", "content": prompt})

                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        if not st.session_state.gemini_model:
                            st.error("Please configure your Gemini API key first")
                        else:
                            try:
                                # Create a context-aware prompt
                                full_prompt = (
                                    f"Document content:\n{st.session_state.file_content[:10000]}\n\n"
                                    f"Question: {prompt}\n"
                                    "Answer based on the document content:"
                                )

                                response = st.session_state.gemini_model.generate_content(full_prompt)
                                st.markdown(response.text)
                                st.session_state.file_messages.append({
                                    "role": "assistant",
                                    "content": response.text
                                })
                            except Exception as e:
                                st.error(f"Error generating response: {str(e)}")

        finally:
            # Ensure the temporary file is closed and deleted
            try:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            except Exception as e:
                st.error(f"Error cleaning up temporary file: {str(e)}")


def chat_with_search_interface():
    st.title("Chat with Web Search")
    st.caption("Conversation augmented with web search results")

    if "search_messages" not in st.session_state:
        st.session_state.search_messages = []

    for message in st.session_state.search_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    col1, col2 = st.columns([4, 1])
    with col1:
        prompt = st.chat_input("Your message with web context...")
    with col2:
        search_web = st.checkbox("Enable web search", value=True)

    if prompt:
        st.session_state.search_messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not st.session_state.gemini_model:
                st.error("Please configure your Gemini API key first")
            else:
                try:
                    if search_web:
                        prompt_with_context = f"Perform a web search if needed to answer: {prompt}"
                    else:
                        prompt_with_context = prompt

                    response = st.session_state.gemini_model.generate_content(prompt_with_context)
                    st.markdown(response.text)
                    st.session_state.search_messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")


def langchain_quickstart():
    st.title("LangChain Quickstart with Gemini")
    st.caption("Basic LangChain integration examples")

    st.info("""
    This section demonstrates basic LangChain patterns using Gemini.
    Note: This is a mock implementation showing how you would integrate LangChain with Gemini.
    """)

    example = st.selectbox("Select example", [
        "Simple Chain",
        "Prompt Template",
        "Memory Conversation",
        "Document Loader"
    ])

    if example == "Simple Chain":
        st.code("""
        from langchain_google_genai import GoogleGenerativeAI
        from langchain_core.prompts import PromptTemplate

        llm = GoogleGenerativeAI(model="gemini-pro", google_api_key="YOUR_KEY")

        prompt = "Tell me a joke about {topic}"
        chain = PromptTemplate.from_template(prompt) | llm

        response = chain.invoke({"topic": "programming"})
        print(response)
        """)

    elif example == "Prompt Template":
        st.code("""
        from langchain_google_genai import GoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate

        llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.7)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant"),
            ("user", "{input}")
        ])

        chain = prompt | llm
        response = chain.invoke({"input": "Explain quantum computing"})
        """)

    elif example == "Memory Conversation":
        st.code("""
        from langchain_google_genai import GoogleGenerativeAI
        from langchain.memory import ConversationBufferMemory
        from langchain.chains import ConversationChain

        llm = GoogleGenerativeAI(model="gemini-pro")
        memory = ConversationBufferMemory()

        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=True
        )

        conversation.predict(input="Hi there!")
        conversation.predict(input="Tell me about yourself")
        """)

    elif example == "Document Loader":
        st.code("""
        from langchain_google_genai import GoogleGenerativeAI
        from langchain.document_loaders import TextLoader
        from langchain.indexes import VectorstoreIndexCreator

        loader = TextLoader("document.txt")
        index = VectorstoreIndexCreator().from_loaders([loader])

        llm = GoogleGenerativeAI(model="gemini-pro")

        query = "What is the main topic of this document?"
        answer = index.query(query, llm=llm)
        """)


def langchain_prompttemplate():
    st.title("LangChain Prompt Templates")
    st.caption("Create and test prompt templates for Gemini")

    st.info("""
    This section allows you to create and test LangChain prompt templates with Gemini.
    """)

    template = st.text_area(
        "Enter your prompt template (use {variables})",
        """You are an expert in {field}. Explain {concept} in simple terms for a {audience} audience."""
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        field = st.text_input("field", "technology")
    with col2:
        concept = st.text_input("concept", "large language models")
    with col3:
        audience = st.text_input("audience", "5th grade student")

    if st.button("Test Template"):
        if not st.session_state.gemini_model:
            st.error("Please configure your Gemini API key first")
        else:
            try:
                filled_prompt = template.format(field=field, concept=concept, audience=audience)

                st.subheader("Generated Prompt")
                st.code(filled_prompt, language="text")

                st.subheader("Gemini Response")
                with st.spinner("Generating response..."):
                    response = st.session_state.gemini_model.generate_content(filled_prompt)
                    st.markdown(response.text)
            except KeyError as e:
                st.error(f"Missing template variable: {str(e)}")
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")


def chat_with_feedback():
    st.title("Chat with User Feedback")
    st.caption("Conversation with built-in feedback mechanism")

    if "feedback_messages" not in st.session_state:
        st.session_state.feedback_messages = []

    for message in st.session_state.feedback_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "feedback" in message:
                st.caption(f"Feedback: {message['feedback']}")

    if prompt := st.chat_input("Your message..."):
        st.session_state.feedback_messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not st.session_state.gemini_model:
                st.error("Please configure your Gemini API key first")
            else:
                try:
                    response = st.session_state.gemini_model.generate_content(prompt)
                    st.markdown(response.text)

                    msg = {"role": "assistant", "content": response.text}
                    st.session_state.feedback_messages.append(msg)

                    # Feedback buttons
                    col1, col2, col3 = st.columns([1, 1, 5])
                    with col1:
                        if st.button("👍", key=f"thumbs_up_{len(st.session_state.feedback_messages)}"):
                            msg["feedback"] = "positive"
                    with col2:
                        if st.button("👎", key=f"thumbs_down_{len(st.session_state.feedback_messages)}"):
                            msg["feedback"] = "negative"
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")


# Main app router
if app_mode == "Chat":
    chat_interface()
elif app_mode == "File Q&A":
    file_qa_interface()
elif app_mode == "Chat with Search":
    chat_with_search_interface()
elif app_mode == "Langchain Quickstart":
    langchain_quickstart()
elif app_mode == "Langchain PromptTemplate":
    langchain_prompttemplate()
elif app_mode == "Chat with Feedback":
    chat_with_feedback()