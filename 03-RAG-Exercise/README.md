# RAG Exercise with Milvus & Langfuse

This exercise demonstrates how to build a Retrieval-Augmented Generation (RAG) system using:
- **Milvus** as the vector database for embeddings.
- **OpenAI** for embeddings and chat completions.
- **Langfuse** to log all API calls.
- **Streamlit** as a chat UI.

## Directory Structure

```
rag_milvus_langfuse_exercise/
│
├── rag_milvus_langfuse.py      ← Streamlit app implementing RAG
├── requirements.txt            ← Python dependencies
├── README.md                   ← Overview
├── INSTRUCTIONS.md             ← Detailed instructions
└── pdfs/
    └── [Place your PDF files here]
```

## Setup

1. **Ensure Milvus is running locally**  
   - Setup a Milvius account on zilliz

2. **Install Python dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your PDF documents**  
   - Copy your `.pdf` files into the `pdfs/` directory.

4. **Run the Streamlit app**  
   ```bash
   streamlit run rag.py
   ```

5. **Interact with the Chat UI**  
   - Ask questions about the PDFs in the chat input.
   - The system retrieves relevant text chunks via Milvus and answers using OpenAI.  
   - All API calls (embeddings & chat) are logged in Langfuse.

## Next Steps
- Explore the Langfuse dashboard to inspect logged runs (`run_type: "embedding"` or `"chat"`).  
- Adjust chunk size or number of search hits (`limit`) for better retrieval.  
- Experiment with different OpenAI models or Milvus index parameters.  
