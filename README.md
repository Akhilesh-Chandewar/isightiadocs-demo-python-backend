# NLP Document Interaction Server

This is a Flask server for an NLP (Natural Language Processing) application that enables users to interact with documents using natural language to perform various document-related tasks. The server utilizes the LangChain library for text processing, embeddings, and conversational retrieval.

## Requirements
- Python 3.x
- Flask
- Flask-Limiter
- LangChain
- OpenAI GPT-3.5-turbo (model not included, must be configured separately)

## Setup

1. Install dependencies:
    ```bash
    pip install flask flask-limiter langchain openai
    ```

2. Configure OpenAI API key: Set your OpenAI GPT-3.5-turbo API key in the `.env` file.

3. Run the server:
    ```bash
    python your_file_name.py
    ```

## Endpoints

### 1. `/`
- **Method:** GET
- **Description:** Home endpoint providing a simple welcome message.
- **Example Response:**
    ```json
    {
        "message": "Hello, this is a Flask server for NLP application that allows users to interact with documents using natural language to perform various document-related tasks."
    }
    ```

### 2. `/process`
- **Method:** POST
- **Description:** Process a CSV file and set up the conversation chain.
- **Request:**
    - Form Data:
        - `csv_file`: CSV file containing text data.
- **Example Response:**
    ```json
    {
        "status": "success"
    }
    ```

### 3. `/ask_question`
- **Method:** POST
- **Description:** Ask a question related to the processed documents.
- **Request:**
    - JSON Data:
        - `user_question`: User's question.
- **Example Response:**
    ```json
    {
        "chat_history": "The generated response from the conversation chain."
    }
    ```

## Implementation Details

### 1. `get_csv_text(csv_file)`
- Reads a CSV file and returns concatenated text from the 'text' column.

### 2. `get_text_chunks(text)`
- Splits the input text into chunks using LangChain's `CharacterTextSplitter`.

### 3. `get_vectorstore(text_chunks)`
- Generates embeddings for text chunks using OpenAI's GPT-3.5-turbo and creates a vector store with FAISS.

### 4. `get_conversation_chain(vectorstore)`
- Sets up a conversational retrieval chain using LangChain with a ChatOpenAI language model, the vector store, and a conversation buffer memory.

### 5. `/process` Endpoint
- Accepts a CSV file, processes it, and initializes the conversation chain.

### 6. `/ask_question` Endpoint
- Accepts a user's question, queries the conversation chain, and returns the chat history.

## Rate Limiting
- Rate limiting is applied to both `/process` and `/ask_question` endpoints to ensure the responsible use of resources. Adjust the limits based on your OpenAI model's capabilities.

## Note
- This code assumes that the LangChain library is properly installed and configured, along with the necessary dependencies. Make sure to adapt the code based on your specific environment and requirements.
