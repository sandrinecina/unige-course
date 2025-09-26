#
# os, time: File system, delays.
# streamlit: For web UI.
# pdfplumber: For extracting text from PDFs.
# openai: OpenAI’s Python SDK (for embeddings and chat).
# langfuse: Logs all interactions for analytics/tracing.
# pymilvus: For vector DB operations (Milvus).
# pandas: For data handling if needed.
# rich.console: For optional pretty-printing in the backend.
#
import os
import streamlit as st
import pdfplumber
#from openai import OpenAI
from langfuse import Langfuse
from langfuse.openai import openai as openai_client
from pymilvus import MilvusClient, FieldSchema, DataType, Collection, CollectionSchema, connections, utility
import pandas as pd
from rich.console import Console
import uuid

#
# API and DB Client Configuration
# Sets up OpenAI and Langfuse clients using your API keys.
# Connects to your Milvus vector database using the provided URI/token.
#
#openai_client = OpenAI()

# Initialize Langfuse client
lf_client = Langfuse(
  secret_key=os.getenv("LF_SECRET_KEY"),
  public_key=os.getenv("LF_PUBLIC_KEY"),
  host="https://cloud.langfuse.com"
)

# Milvus connection parameters
milvus_uri = os.getenv("MILVUS_URI")
milvus_token = os.getenv("MILVUS_KEY")
connections.connect(uri=milvus_uri, token=milvus_token, timeout=30)
print(f"Connected to DB: {milvus_uri} successfully")

#
# Milvus Collection Details
# Collection name for vector storage.
# Embedding dimension matches OpenAI’s default for embedding models.
#
COLLECTION_NAME = "llm_papers_collection_v2"
EMBED_DIM = 1536  # OpenAI embedding dimension

console = Console()

#
# Helper Functions
#

#
# call_embedding_and_log
# Sends text to OpenAI to get a vector embedding.
# Logs the embedding call to Langfuse (input text, first few values of the embedding, tokens used, metadata).
# Returns the embedding vector for later indexing/search.
#
def call_embedding_and_log(text: str, sample_id: str):
    """
    Call OpenAI embeddings API and log to Langfuse.
    """
    try:
        response = openai_client.embeddings.create(
            name="Embedding",
            model="text-embedding-3-small",
            input=text, 
            metadata={"provider": "openai", 
                      "model": "text-embedding-3-small", 
                      "langfuse_tags": ["streamlit", "prompt-engineering", "example"], 
                      "langfuse_session_id":str(uuid.uuid4()), 
                      "langfuse_user_id":str(uuid.uuid4()),
                      "sample_id": sample_id,
                      }
            )
    except Exception as e:
        console.print(f"[red]Langfuse logging error (embedding):[/red] {e}")
    return response.data[0].embedding


#
# call_chat_and_log
# Calls OpenAI ChatCompletion with a context-augmented prompt.
# Logs prompt, answer, tokens, and sample ID to Langfuse.
#
def call_chat_and_log(prompt: str, context: str, sample_id: str):
    """
    Call OpenAI chat completion with context and log to Langfuse.
    """
    full_prompt = f"""Use the following context to answer the question: {context} 
    Question: {prompt}"""

    try:
        response = openai_client.chat.completions.create(
        name="chat",                     # shows up as trace/generation name in Langfuse
        model="gpt-4.1",                 # keep your model; use any you like
        messages=[{"role": "user", "content": full_prompt}],
        # The following extras are parsed by Langfuse and attached to the trace:
        metadata={"provider": "openai", 
                  "model": "gpt-4.1", 
                  "langfuse_tags": ["streamlit", "RAG", "example"], 
                  "langfuse_session_id":str(uuid.uuid4()), 
                  "langfuse_user_id":str(uuid.uuid4()),
                  "sample_id": sample_id,
                  "run_type": "chat"},
        )
    except Exception as e:
        console.print(f"[red]Langfuse logging error (chat):[/red] {e}")
    return response.choices[0].message.content.strip()

#
# initialize_milvus
# Checks if the collection exists; if not, creates it with fields for embedding, chunk ID, and original text.
#
def initialize_milvus():
    client = MilvusClient(uri=milvus_uri, token=milvus_token)

    if client.has_collection(COLLECTION_NAME):
        client.load_collection(COLLECTION_NAME)
        return client

    # 1) build schema
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)                 # auto_id comes from schema
    schema.add_field("chunkid", DataType.VARCHAR, max_length=65535)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBED_DIM)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)

    # 2) index params
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="L2",
        params={"nlist": 128},
    )

    # 3) create + load
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    client.load_collection(COLLECTION_NAME)
    return client

#
# load_and_index_pdfs
# Scans the pdfs/ directory for PDFs.
# Extracts text from each page.
# Chunks text (e.g., every 1000 chars).
# Embeds each chunk and inserts the embedding, text, and IDs into Milvus for search.
#
def load_and_index_pdfs(milvius_client):
    """
    Load PDFs from 'pdfs/' directory, extract text, chunk, embed, and insert into Milvus.
    """
    docs = []
    for filename in os.listdir("pdfs"):
        if filename.lower().endswith(".pdf"):
            path = os.path.join("pdfs", filename)
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        docs.append((f"{filename}_page_{i}", text))
    chunks = []
    for doc_id, text in docs:
        for i in range(0, len(text), 1000):
            chunk_text = text[i:i+1000]
            chunks.append((f"{doc_id}_chunk_{i}", chunk_text))
    data=[]
    for sample_id, chunk_text in chunks:
        embedding = call_embedding_and_log(chunk_text, sample_id)
        data.append({"chunkid": sample_id, 
                     "embedding": embedding, 
                     "text": chunk_text})
    milvius_client.insert(COLLECTION_NAME, data)
    #collection.flush()

#
# Streamlit App
# App Layout and Initialization
# Streamlit app title and config.
# Collection is initialized; if empty, PDFs are indexed automatically.
#
st.set_page_config(page_title="RAG with Milvus & Langfuse", layout="wide")
st.title("📚 Retrieval-Augmented Generation (RAG) Chat")

milvius_client = initialize_milvus()
print("collection", milvius_client.get_collection_stats(COLLECTION_NAME))
count = milvius_client.get_collection_stats(COLLECTION_NAME)["row_count"]
print("count=", count)
if count == 0:
    st.info("Indexing PDFs into Milvus... This may take a few minutes.")
    load_and_index_pdfs(milvius_client)
    st.success("Indexing complete!")

if "messages" not in st.session_state:
    st.session_state.messages = []

#
# Sidebar Instructions
# Easy guide for the user.
#
with st.sidebar:
    st.header("Instructions")
    st.write("""
    1. Place your PDF files in the `pdfs/` directory.
    2. Ensure Milvus connectivity.
    3. Enter your question in the chat box below.
    4. The system retrieves relevant excerpts and answers using OpenAI.
    5. All API calls (embeddings & chat) are logged in Langfuse.
    """)

#
# Message Loop and Retrieval
# Chat UI: Displays the full conversation.
# On user input:
# Embeds the question.
# Searches Milvus for the 3 most relevant chunks.
# Assembles context from those chunks.
# Calls OpenAI with context and the user's question.
# Logs everything to Langfuse.
# Adds the answer to the chat display.
#
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

user_input = st.chat_input("Ask a question about the PDFs...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    query_emb = call_embedding_and_log(user_input, "query")

    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    result = milvius_client.search(COLLECTION_NAME,data=[query_emb],search_params=search_params,limit=3,output_fields=["text"])
    context = ""
    for hits in result:
        print(hits.ids)
        for hit in hits:
            context += hit.entity.get('text') + "\n\n -- \n\n"

    answer = call_chat_and_log(user_input, context, "chat_request")
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").markdown(answer)

#
# Summary: What does this app do?
#
# Indexes PDF documents using OpenAI embeddings and Milvus vector DB.
# Lets the user query any document through a chat UI.
# Retrieves the most relevant text segments as context and passes them to OpenAI for an answer (RAG).
# All embedding and chat requests are logged to Langfuse for audit, observability, and prompt improvement.
#
