
<!-- Header Section -->
    
<h1>Q&A Chatbot System Flow</h1>
<p>Explaining how the chatbot interacts with Google Generative AI API and presents the chat interface using Streamlit.</p>

<!-- Diagram Section -->
<h2>Live Demo>
    https://chatbotgemini-by-shuvankar.streamlit.app/
<h2>System Flow Diagram</h2>

<!-- User Interface Image -->

<h3>1. User Interface (Streamlit Input)</h3>
<img src="interface.png" alt="User Interface Input">
<p>The user enters a question in the Streamlit app.</p>


<!-- API Integration Image -->
<h3>2. Google Generative AI API Integration</h3>
<img src="flow.png" alt="API Integration">
<a>Not Available </a>
<p>The question is sent to Google Generative AI (Gemini Model) for processing.</p>

<!-- Response and Chat History Image -->
<h3>3. Bot Response and Chat History</h3>
<img src="hist.png" alt="Bot Response and Chat History">
<p>The response is displayed in the chat interface, and the conversation history is maintained.</p>

<!-- How It Works Section -->

<h2>How It Works</h2>
<p>Here's how the chatbot system works in three stages:</p>

<h3>Step 1: User Input</h3>
<p>The user types their question in the Streamlit app’s input box and submits it by pressing the "Ask" button.</p>

<h3>Step 2: API Processing</h3>
<p>The input is sent to the Google Generative AI API (Gemini Model), where the question is processed, and a response is generated.</p>

<h3>Step 3: Response Display</h3>
<p>The generated response is displayed in the chat interface, and the conversation is stored in the chat history using Streamlit's session state.</p>

