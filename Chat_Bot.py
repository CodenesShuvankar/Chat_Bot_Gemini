import streamlit as st
import google.generativeai as genai
from PIL import Image
import fitz  # PyMuPDF
import docx
import tempfile
import os
import json

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

    st.link_button("Get Gemini API Key", "https://aistudio.google.com/app/apikey")

    # Navigation
    st.subheader("Navigation")
    app_mode = st.radio(
        "Select Mode",
        ["Chat", "File Q&A", "Chat with Search", "Agentic AI", "Langchain Quickstart",
         "Langchain PromptTemplate", "Chat with Feedback"],
        index=0
    )

    # Source code link
    st.divider()
    st.subheader("About")
    st.link_button("View Source Code", "https://github.com/CodenesShuvankar/Chat_Bot_Gemini")
    st.link_button("Open in GitHub Codespaces", "https://github.com/CodenesShuvankar/Chat_Bot_Gemini")


# Main app functions
def chat_interface():
    st.title("Gemini Chat")
    st.caption("A conversational AI powered by Google Gemini")

    # Initialize chat history
    def _load_chat_history():
        try:
            if os.path.exists("chat_history.json"):
                with open("chat_history.json", "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    def _save_chat_history():
        try:
            with open("chat_history.json", "w", encoding="utf-8") as f:
                json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.warning(f"Couldn't save chat history: {e}")

    if "messages" not in st.session_state:
        st.session_state.messages = _load_chat_history()
        # If API is configured, rebuild chat session from history
        try:
            if st.session_state.gemini_model and st.session_state.messages:
                history = []
                for m in st.session_state.messages:
                    role = m.get("role")
                    content = m.get("content", "")
                    if role == "assistant":
                        role = "model"
                    history.append({"role": role, "parts": [content]})
                st.session_state.chat_session = st.session_state.gemini_model.start_chat(history=history)
        except Exception:
            pass

    # Controls
    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("Clear chat history"):
            st.session_state.messages = []
            _save_chat_history()
            # Reset the underlying Gemini chat session as well
            try:
                if st.session_state.gemini_model:
                    st.session_state.chat_session = st.session_state.gemini_model.start_chat(history=[])
            except Exception:
                pass
            st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Your message"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        _save_chat_history()

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
                    _save_chat_history()
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


def agentic_ai_interface():
    st.title("Agentic AI (Tools + Planning)")
    st.caption("An agent that can plan and use tools: web search, calculator, and notes memory.")
    import json as _json  # local alias to avoid analyzer scope issues

    if not st.session_state.gemini_model:
        st.warning("Please configure your Gemini API key in the sidebar first.")
        return

    # Tool toggles
    with st.expander("Tools configuration"):
        enable_search = st.checkbox("Enable Web Search", value=True)
        enable_calc = st.checkbox("Enable Calculator", value=True)
        enable_memory = st.checkbox("Enable Notes Memory", value=True)
        max_steps = st.slider("Max reasoning/tool steps", 1, 6, 3)

    # Session state for agent
    if "agent_messages" not in st.session_state:
        # Load previous agent history if exists
        try:
            if os.path.exists("agent_history.json"):
                with open("agent_history.json", "r", encoding="utf-8") as f:
                    st.session_state.agent_messages = _json.load(f)
            else:
                st.session_state.agent_messages = []
        except Exception:
            st.session_state.agent_messages = []

    # Display conversation
    for message in st.session_state.agent_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Controls
    colx, coly = st.columns([1,1])
    with colx:
        if st.button("Clear agent history"):
            st.session_state.agent_messages = []
            try:
                with open("agent_history.json", "w", encoding="utf-8") as f:
                    _json.dump([], f)
            except Exception:
                pass
            st.rerun()

    # Build tool declarations for function calling
    def get_tool_declarations():
        decls = []
        if enable_search:
            decls.append({
                "name": "web_search",
                "description": "Search the web for recent information and return a concise list of results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer"}
                    },
                    "required": ["query"]
                }
            })
        if enable_calc:
            decls.append({
                "name": "calculator",
                "description": "Evaluate a basic math expression with + - * / ** // % and parentheses.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    },
                    "required": ["expression"]
                }
            })
        if enable_memory:
            decls.append({
                "name": "write_note",
                "description": "Store a short note by key for later recall.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["key", "content"]
                }
            })
            decls.append({
                "name": "read_note",
                "description": "Retrieve a previously stored note by key.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"}
                    },
                    "required": ["key"]
                }
            })
        return decls

    # Whitelist of official music platforms to avoid copyright-violating links
    MUSIC_WHITELIST = {
        "youtube.com", "www.youtube.com", "music.youtube.com",
        "open.spotify.com", "music.apple.com", "music.amazon.com",
        "gaana.com", "www.jiosaavn.com", "wynk.in", "soundcloud.com"
    }

    # Detect intents that could lead to copyright infringement
    def is_music_download_intent(text: str) -> bool:
        t = (text or "").lower()
        return ("download" in t or "mp3" in t) and ("song" in t or "music" in t)

    def load_memory_file():
        try:
            import json
            if os.path.exists("agent_memory.json"):
                with open("agent_memory.json", "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def save_memory_file(data):
        try:
            import json
            with open("agent_memory.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"Failed saving memory: {e}")

    def run_tool(name, args):
        try:
            if name == "web_search":
                try:
                    from ddgs import DDGS  # new package name
                except Exception:
                    from duckduckgo_search import DDGS  # fallback for older envs
                from urllib.parse import urlparse
                q = args.get("query", "")
                n = int(args.get("max_results", 3) or 3)
                results = []
                with DDGS() as ddgs:
                    for r in ddgs.text(q, max_results=n):
                        url = r.get("href") or r.get("url")
                        title = r.get("title")
                        snippet = r.get("body") or r.get("snippet")
                        # Filter suspicious direct file links or shady download sites
                        try:
                            parsed = urlparse(url)
                            host = (parsed.netloc or "").lower()
                            path = (parsed.path or "").lower()
                            is_audio = any(path.endswith(ext) for ext in [
                                ".mp3", ".m4a", ".aac", ".flac", ".wav", ".ogg"
                            ])
                            is_suspicious_download = ("download" in host or "download" in path)
                            # If intent looks like music download, enforce whitelist
                            if is_music_download_intent(q):
                                if host not in MUSIC_WHITELIST:
                                    continue
                                if is_audio:
                                    continue
                            else:
                                if is_audio and host not in MUSIC_WHITELIST:
                                    continue
                        except Exception:
                            pass
                        results.append({"title": title, "snippet": snippet, "url": url})
                return {"results": results}
            if name == "calculator":
                import ast, operator as op
                allowed_ops = {
                    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
                    ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
                    ast.UAdd: op.pos, ast.USub: op.neg
                }
                def eval_(node):
                    if isinstance(node, ast.Num):
                        return node.n
                    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                        return node.value
                    if isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
                        return allowed_ops[type(node.op)](eval_(node.left), eval_(node.right))
                    if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_ops:
                        return allowed_ops[type(node.op)](eval_(node.operand))
                    raise ValueError("Unsupported expression")
                expr = str(args.get("expression", "")).strip()
                result = eval_(ast.parse(expr, mode="eval").body)
                return {"result": result}
            if name == "write_note":
                store = load_memory_file()
                key = str(args.get("key", "")).strip()
                content = str(args.get("content", ""))
                if not key:
                    raise ValueError("key required")
                store[key] = content
                save_memory_file(store)
                return {"ok": True}
            if name == "read_note":
                store = load_memory_file()
                key = str(args.get("key", "")).strip()
                return {"content": store.get(key)}
        except Exception as e:
            return {"error": str(e)}

    # Run an agent loop using Gemini function calling if available, else fallback to a simple response
    def run_agent(prompt_text: str):
        # Build a fresh model with tool declarations
        tools = [{"function_declarations": get_tool_declarations()}]
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            tools=tools,
            system_instruction=(
                "You are an autonomous assistant that plans steps and uses tools when helpful. "
                "Use prior conversation context; do not ask for info already given—infer from recent turns. "
                "Think step-by-step. If a tool is useful, issue a single function_call with clear arguments. "
                "After tool results return, synthesize a concise, direct answer for the user."
            ),
        )
        # Build chat history from prior agent turns so follow-ups keep context
        history = []
        try:
            prior = st.session_state.get("agent_messages", [])
            # Use the last 12 messages to keep context bounded
            for m in prior[-12:]:
                role = m.get("role")
                content = m.get("content", "")
                if role == "assistant":
                    role = "model"
                history.append({"role": role, "parts": [content]})
        except Exception:
            history = []
        chat = model.start_chat(history=history)

        steps_taken = 0
        final_answer = None
        tool_trace = []

        try:
            response = chat.send_message(prompt_text)
            while steps_taken < max_steps:
                steps_taken += 1
                # Inspect parts for function calls
                parts = []
                try:
                    cand = response.candidates[0]
                    parts = getattr(cand.content, "parts", []) or []
                except Exception:
                    parts = []

                calls = []
                for p in parts:
                    fc = getattr(p, "function_call", None) or getattr(p, "functionCall", None)
                    if fc:
                        try:
                            calls.append({
                                "name": getattr(fc, "name", None),
                                "args": dict(getattr(fc, "args", {}) or {})
                            })
                        except Exception:
                            pass

                if not calls:
                    # No tool calls -> treat as final
                    final_answer = getattr(response, "text", None) or ""
                    break

                # Execute at most one tool per step (first call)
                call = calls[0]
                result = run_tool(call.get("name"), call.get("args") or {})
                tool_trace.append({"tool": call.get("name"), "args": call.get("args"), "result": result})

                # Send tool result back
                try:
                    response = chat.send_message([
                        {
                            "function_response": {
                                "name": call.get("name"),
                                "response": result,
                            }
                        }
                    ])
                except Exception:
                    # Fallback: append as plain text context
                    response = chat.send_message(
                        f"Tool {call.get('name')} result: {result}. Now provide the next step or final answer.")

            if final_answer is None:
                final_answer = getattr(response, "text", None) or ""

        except Exception as e:
            final_answer = f"Agent failed: {e}"
            tool_trace.append({"error": str(e)})

        return final_answer, tool_trace

    # Chat input
    if user_input := st.chat_input("Ask me to accomplish a task…"):
        st.session_state.agent_messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Thinking and using tools…"):
                answer, trace = run_agent(user_input)
                st.markdown(answer)
                if st.checkbox("Show tool trace"):
                    import json
                    st.code(json.dumps(trace, ensure_ascii=False, indent=2), language="json")

        st.session_state.agent_messages.append({"role": "assistant", "content": answer})
        try:
            with open("agent_history.json", "w", encoding="utf-8") as f:
                _json.dump(st.session_state.agent_messages, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


# Main app router
if app_mode == "Chat":
    chat_interface()
elif app_mode == "File Q&A":
    file_qa_interface()
elif app_mode == "Chat with Search":
    chat_with_search_interface()
elif app_mode == "Agentic AI":
    agentic_ai_interface()
elif app_mode == "Langchain Quickstart":
    langchain_quickstart()
elif app_mode == "Langchain PromptTemplate":
    langchain_prompttemplate()
elif app_mode == "Chat with Feedback":
    chat_with_feedback()