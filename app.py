import pandas as pd
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

app = Flask(__name__)

limiter = Limiter(get_remote_address,app=app ,storage_uri="memory://")

conversation_chain = None

def get_csv_text(csv_file):
    df = pd.read_csv(csv_file)
    text = ' '.join(df['text'].astype(str))
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

@app.route('/')
def home():
    return "Hello this is flask server for NLP application that allows users to interact with documents using natural language to perform various document related task"

@app.route('/process', methods=['POST'])
@limiter.limit("1 per day")  # Adjust the rate limit based on your model (gpt-3.5-turbo)
def process_csv():
    global conversation_chain
    csv_file = request.files['csv_file']
    raw_text = get_csv_text(csv_file)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore)
    return jsonify({"status": "success"})

@app.route('/ask_question', methods=['POST'])
@limiter.limit("1 per day")
def ask_question():
    user_question = request.json.get('user_question')
    response = conversation_chain({'question': user_question})
    chat_history = response['chat_history']
    return jsonify({"chat_history": chat_history})

if __name__ == '__main__':
    load_dotenv()
    app.run(debug=True)