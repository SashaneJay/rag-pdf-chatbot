import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

st.set_page_config(page_title="AI PDF Chatbot")

st.title("📄 AI PDF Chatbot")

st.write(
    "Upload a PDF document and ask questions about its contents."
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
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
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)
    chunk_embeddings = embedding_model.encode(
    chunks,
    show_progress_bar=False
    )

    chunk_embeddings = np.array(
        chunk_embeddings,
        dtype=np.float32
    )

    dimension = chunk_embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(chunk_embeddings)
    
    st.subheader("Document Statistics")

    st.write("Pages:", len(pdf_reader.pages))
    st.write("Characters:", len(text))
    st.write("Chunks:", len(chunks))

    st.write(
    "Embedding Shape:",
    chunk_embeddings.shape
    )
    
    st.write(
    "Vectors Stored:",
    index.ntotal
    )

    with st.expander("Preview Extracted Text"):
        st.write(text[:3000])

    with st.expander("Preview First Chunk"):
        st.write(chunks[0])

    question = st.text_input(
        "Ask a question about the document"
    )

    if question:

        question_embedding = embedding_model.encode(
            [question]
        )

        question_embedding = np.array(
            question_embedding,
            dtype=np.float32
        )

        distances, indices = index.search(
            question_embedding,
            k=3
        )

        retrieved_chunks = [
            chunks[idx]
            for idx in indices[0]
        ]

        answer_context = "\n\n".join(retrieved_chunks)

        st.subheader("Answer")

        st.write(
            "Based on the most relevant parts of the document:"
        )

        st.info(answer_context[:1500])

        with st.expander("Show Retrieved Chunks"):
            for rank, idx in enumerate(indices[0], start=1):
                st.markdown(f"### Match {rank}")
                st.write(chunks[idx])
                st.write(f"Distance: {distances[0][rank-1]:.4f}")