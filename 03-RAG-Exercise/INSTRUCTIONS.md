# RAG with Milvus & Langfuse – Detailed Instructions

## 1. Prerequisites

- **Milvus** Use Milvius on zilliz.
- **Python 3.9+** environment.
- API keys for **OpenAI** and **Langfuse**.

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `streamlit` – for the chat UI.
- `pdfplumber` – to extract text from PDFs.
- `openai>=0.27.0` – for embeddings and chat.
- `langfuse>=0.1.0` – for logging.
- `pymilvus` – to interact with Milvus.
- `pandas`, `rich` – utility libraries.

## 3. Configuration

1. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY="sk-..."
   export LANGFUSE_API_KEY="lf-..."
   export LANGFUSE_WORKSPACE_ID="your_workspace_id"
   ```
2. **Ensure Milvus is running**:
   - Default connection: `host=localhost`, `port=19530`
   - If using Docker:
     ```bash
     docker run -d --name milvus-standalone -p 19530:19530 -p 19121:19121 milvusdb/milvus:v2.2.5-android-arm64
     ```

## 4. Prepare PDFs

- Place any PDF files you want to query under the `pdfs/` directory.
- The script will automatically load all `.pdf` files, extract text by page, chunk it into 1000-character segments, compute embeddings, and index into Milvus.

## 5. Understanding the Script (`rag_milvus_langfuse.py`)

### 5.1 Initialization

- **OpenAI Client**:
  ```python
  openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
  ```
- **Langfuse Client**:
  ```python
  lf_client = LangfuseClient(api_key=LF_API_KEY, workspace_id=LF_WORKSPACE_ID)
  ```

- **Milvus Connection**:
  ```python
  connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
  ```
- Checks if `rag_collection` exists; if not, creates with schema:
  - `id` (auto-increment primary key)
  - `embedding` (FLOAT_VECTOR, dim=1536)
  - `text` (VARCHAR)

### 5.2 Loading & Indexing PDFs

1. **Extract Text**:
   - Uses `pdfplumber` to open each PDF and extract page text.
2. **Chunking**:
   - Splits each page’s text into 1000-character chunks.
3. **Embedding**:
   - For each chunk, calls OpenAI embeddings (`text-embedding-3-embedding-ada-002`) and logs the call to Langfuse with `run_type="embedding"`.
4. **Insert into Milvus**:
   - Inserts `(chunk_id, embedding_vector, text_chunk)` for all chunks.  
   - Creates or uses an IVF_FLAT index for efficient search, then loads the collection.

### 5.3 Chat UI (Streamlit)

- On first run, if `collection.num_entities == 0`, shows “Indexing PDFs…” message while indexing.
- Maintains a chat history in `st.session_state.messages`.
- On user input:
  1. **Embed Query**:  
     ```python
     query_emb = call_embedding_and_log(user_input, "query")
     ```  
     - Logs to Langfuse with `run_type="embedding"`, `sample_id="query"`.
  2. **Milvus Search**:  
     ```python
     results = collection.search(
         data=[query_emb],
         anns_field="embedding",
         param={"metric_type": "L2", "params": {"nprobe": 10}},
         limit=3,
         output_fields=["text"]
     )
     ```  
     - Retrieves top-3 most similar chunks.
  3. **Assemble Context**:  
     - Concatenate retrieved chunks into a single context string.
  4. **Call Chat**:  
     ```python
     answer = call_chat_and_log(user_input, context, "chat_request")
     ```  
     - Logs to Langfuse with `run_type="chat"`, `sample_id="chat_request"`.
  5. **Display Answer**:  
     - Appends assistant message to chat history.

## 6. Running the App

```bash
streamlit run rag_milvus_langfuse.py
```

- Wait for “Indexing PDFs…” to finish if first run.
- Ask questions in the chat UI about your PDF content.
- View logs in Langfuse under runs tagged `run_type="embedding"` and `run_type="chat"`.

## 7. Post-Run Analysis

1. **Langfuse Dashboard**:
   - Filter by `tags.run_type="chat"` to inspect question-answer pairs and token usage.
   - Filter by `tags.run_type="embedding"` to inspect embedding calls.
2. **Milvus Collection**:
   - Use Milvus SDK to check number of entities:
     ```python
     from pymilvus import Collection
     collection = Collection("rag_collection")
     print(collection.num_entities)
     ```

## 8. Extensions

- **Adjust Chunk Size**: Modify the 1000-character chunk length for finer or coarser granularity.
- **Increase Retrieval Hits**: Change `limit=3` to a higher number for more context.
- **Model Variations**: Experiment with other OpenAI models for embeddings or chat.
- **Advanced Prompting**: Prepend system messages or use few-shot examples in `full_prompt`.

Enjoy building and querying your RAG system with Milvus and Langfuse!
