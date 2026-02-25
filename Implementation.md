Goal Description
Migrate the existing local Streamlit-based NCERT-RAG-Bot to a production-ready Full-Stack Web Application hosted on Vercel.

The current application relies heavily on local resources that are incompatible with Vercel's ephemeral, serverless environment:

Local File System: Uploads and ChromaDB vector databases are saved to the local disk.
Local LLM: It uses local Ollama for embeddings (nomic-embed-text) and chat (phi3-fast).
To host this on Vercel, we need to decouple these local dependencies and move to a cloud-native architecture.

WARNING

Vercel's serverless functions are ephemeral. They cannot store persistent data on the disk (like ChromaDB or uploaded PDFs) and cannot run heavy background processes like an Ollama server.

Proposed Architecture for Vercel
1. Frontend (UI)
Framework: Next.js (App Router) with React. Next.js is highly optimized for Vercel.
Styling: Tailwind CSS (standard for modern Next.js apps) or Vanilla CSS to recreate the dark-themed "NCERT Tutor" UI.
Features:
Sidebar for PDF uploads and textbook selection.
Main chat interface with message history, styling, and markdown support.
2. Backend API (Serverless Functions)
Framework: Vercel Python Serverless Functions (FastAPI) or Next.js API Routes (Node.js).
Recommendation: Use Python Serverless Functions (api/chat.py, api/upload.py) to reuse most of the existing langchain and pypdf logic.
3. Vector Database (Replacing local ChromaDB)
Problem: ChromaDB saves files to ./ncert_db/. Vercel serverless functions lose these files as soon as the function execution ends.
Solution: Migrate to a Managed Cloud Vector Database.
Options: Pinecone, Supabase (pgvector), Upstash Vector, or Qdrant Cloud.
Recommendation: Pinecone or Upstash Vector as they have generous free tiers and seamless LangChain integration.
4. File Storage for PDFs (Replacing local ./uploaded_pdfs/)
Problem: PDFs saved locally will disappear between API requests.
Solution: Use Cloud Object Storage.
Options: Vercel Blob or AWS S3.
Recommendation: Vercel Blob for native integration and ease of use.
5. LLM and Embeddings (Replacing local Ollama)
Problem: Ollama requires a persistent GPU/CPU server, which Vercel Serverless does not provide.
Solution A (Cloud APIs): Switch to managed APIs like Groq (for ultra-fast Llama-3/Mistral), Together AI, OpenAI, or Google Gemini. (Groq offers a great free tier).
Solution B (Hosted Ollama): If you strictly want to use your custom Ollama models, you must host your Ollama instance on a VPS (like DigitalOcean, AWS EC2, or RunPod) and expose it via an API URL (e.g., OLLAMA_BASE_URL=https://your-server.com). The Vercel app will then make network requests to this external server.
Proposed Changes (File Structure)
If we proceed with a Next.js + Python Serverless approach, the repository structure would look like this:

Frontend Components
[NEW] package.json
Dependencies for Next.js, React, Tailwind CSS, etc.

[NEW] app/page.tsx
Main Chat UI (Input, Message List).

[NEW] app/components/Sidebar.tsx
Sidebar component for uploading and managing PDFs.

Backend (Python Serverless)
[NEW] api/index.py
FastAPI router to handle incoming requests from the frontend.

[MODIFY] 
rag_pipeline.py
 -> api/rag_pipeline.py
Modifications needed:

Remove local ChromaDB references. Initialize the LangChain VectorStore using a cloud provider (e.g., PineconeVectorStore).
Change the ChatOllama and OllamaEmbeddings to use either an external OLLAMA_BASE_URL or a cloud provider like ChatGroq.
[MODIFY] 
data_ingestion.py
 -> api/data_ingestion.py
Update ingestion logic to push embedded chunks to the Cloud Vector DB and upload PDFs to Vercel Blob.

[DELETE] 
streamlit_app.py
The frontend will now be handled by Next.js.

Verification Plan
Since you requested no execution, this plan is for your review. When we execute:

Automated / API Tests
Test PDF uploading to Vercel Blob.
Test text chunking and embedding push to Cloud VectorDB via backend endpoints.
Test the chat endpoint to ensure it retrieves necessary context and returns a streamed/proper response.
Manual Verification
Run Next.js locally (npm run dev) and test the end-to-end chat flow.
Ensure the UI matches or improves upon the original Streamlit dark theme.
Deploy a preview branch to Vercel, configure Environment Variables (Vector DB API Keys, LLM API Keys), and verify the live production build.

Comment
⌥⌘M
