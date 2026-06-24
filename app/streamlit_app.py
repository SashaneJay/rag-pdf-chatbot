import os

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


st.set_page_config(page_title="AI PDF Chatbot")

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    except Exception:
        OPENAI_API_KEY = None

client = None

if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

if not OPENAI_API_KEY and "OPENAI_API_KEY" in st.secrets:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def generate_answer(question, context):
    if client is None:
        return "OpenAI API key not found. Please add it in Streamlit Secrets."
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful RAG assistant. "
                        "Answer using only the provided document context. "
                        "If the answer is not in the context, say you cannot find it in the document."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{question}",
                },
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content

    except Exception as error:
        return f"OpenAI API error: {error}"


embedding_model = load_embedding_model()

st.title("📄 AI PDF Chatbot")

st.write("Upload a PDF document and ask questions about its contents.")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
)

if uploaded_file:
    st.success(f"Loaded: {uploaded_file.name}")

    pdf_reader = PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted + "\n"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = splitter.split_text(text)

    chunk_embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=False,
    )

    chunk_embeddings = np.array(
        chunk_embeddings,
        dtype=np.float32,
    )

    dimension = chunk_embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings)

    st.subheader("Document Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Pages:", len(pdf_reader.pages))
        st.write("Characters:", len(text))
        st.write("Chunks:", len(chunks))

    with col2:
        st.write("Embedding Shape:", chunk_embeddings.shape)
        st.write("Vectors Stored:", index.ntotal)

    with st.expander("Preview Extracted Text"):
        st.write(text[:3000])

    with st.expander("Preview First Chunk"):
        st.write(chunks[0])

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.subheader("Chat")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask a question about the document")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.write(question)

        question_embedding = embedding_model.encode([question])

        question_embedding = np.array(
            question_embedding,
            dtype=np.float32,
        )

        distances, indices = index.search(
            question_embedding,
            k=3,
        )

        retrieved_chunks = [
            chunks[idx]
            for idx in indices[0]
        ]

        answer_context = "\n\n".join(retrieved_chunks)

        if not OPENAI_API_KEY:
            answer = "OpenAI API key not found. Add it to your .env file."
        else:
            answer = generate_answer(
                question,
                answer_context,
            )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.write(answer)

        with st.expander("Show Retrieved Chunks"):
            for rank, idx in enumerate(indices[0], start=1):
                st.markdown(f"### Match {rank}")
                st.write(chunks[idx])
                st.write(f"Distance: {distances[0][rank - 1]:.4f}")

else:
    st.info("Upload a PDF to begin chatting.")