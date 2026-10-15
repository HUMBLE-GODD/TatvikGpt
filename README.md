# ⚡ Tatvikgpt: Azure AI Foundry & Azure Blob Storage Chatbot

A production-grade conversational web application built with **Streamlit**, **Azure AI Foundry**, and **Azure Blob Storage**, styled with an authentic modern dark interface (**Tatvikgpt**) with **ultra-fast native WebSocket streaming (60fps)** and **non-blocking cloud persistence**.

---

## 🌟 Key Features

- **100% Microsoft Azure Stack (Zero AWS / S3 Dependencies)**:
  - **Inference**: Azure AI Foundry (`gpt-5.4-nano`, `gpt-5.4-mini`, `gpt-6-luna`, or custom deployments).
  - **Storage & History**: Azure Blob Storage (stores conversations as JSON blobs under container `chat-history`).
- **Ultra-Fast Performance**:
  - **60fps Native Streaming (`st.write_stream`)**: Direct WebSocket streaming with zero lag, eliminating full-DOM re-renders.
  - **Non-Blocking Background Cloud Sync**: Conversations are saved to local cache in `<1ms` and synced to Azure Blob Storage in background threads (`threading.Thread`).
  - **Instant 0ms History Loading**: Memory-cached metadata renders chat history with zero network latency.
- **Authentic Tatvikgpt Interface**:
  - **Dark Mode**: Dark palette (`#212121` main, `#171717` sidebar, `#2f2f2f` bubbles and input).
  - **Chronological Time Buckets**: History is grouped into **`Today`**, **`Yesterday`**, **`Previous 7 Days`**, **`Previous 30 Days`**, and **`Older`**.
  - **LIFO & 15-Display Limit**: Shows the top 15 most recent chats. When a 16th is created, older chats are moved to an expandable **`Older chats`** section.
  - **Instant Search (`🔍 Search chats...`)**: Filters across *all* conversations (titles, models, and message contents) in real time.
  - **Centered 768px Layout**: Optimal reading and conversational container.
  - **Welcome Landing Screen**: Signature *"What can I help with?"* screen with 4 clickable suggestion chips.
  - **Floating Input Pill**: Rounded floating input bar pinned to the bottom with disclaimer.

---

## 📁 Project Structure

```text
Azure_Foundary/
├── .env                  # Active credentials (Azure API Key & Azure Blob Connection String)
├── .env.example          # Annotated configuration template
├── .gitignore            # Git rules ignoring credentials, cache, and virtual environments
├── .chat_history/        # Local high-speed cache & attachments (git-ignored)
├── app.py                # Main Streamlit web application (heavily documented with comments)
├── storage.py            # Pure Azure Blob Storage engine (heavily documented with comments)
├── requirements.txt      # Python dependencies (Streamlit, OpenAI, Azure Storage Blob, dotenv)
└── README.md             # Documentation and guide
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.9+ installed on your machine.
- An **Azure AI Foundry** project endpoint and API key.
- An **Azure Storage Account** (e.g. `tatvikbot`) with an access key or connection string.

### 2. Activate Virtual Environment

```bash
# macOS / Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configuration in [`.env`](.env)

Open `.env` and verify your Azure credentials:

```ini
# ==============================================================================
# Azure AI Foundry Inference
# ==============================================================================
AZURE_AI_API_KEY=your_azure_api_key
AZURE_AI_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>/openai/v1/responses
AZURE_AI_DEPLOYMENT_NAME=gpt-5.4-nano
AZURE_AI_API_VERSION=2024-08-01-preview
AZURE_AI_ENDPOINT_TYPE=auto

SYSTEM_PROMPT=You are an intelligent, helpful, and concise AI assistant powered by Azure AI Foundry.
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=1024

# ==============================================================================
# Azure Blob Storage Configuration
# ==============================================================================
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=tatvikbot;AccountKey=...;EndpointSuffix=core.windows.net

AZURE_STORAGE_CONTAINER_NAME=chat-history
AZURE_STORAGE_BLOB_PREFIX=chat_history/
```

---

## 🏃 Running the Application

Execute in your terminal:

```bash
./.venv/bin/streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.
