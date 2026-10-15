"""
===============================================================================
AZURE AI FOUNDRY & AZURE BLOB CHATBOT (Tatvikgpt Online UI & Experience)
===============================================================================
A production-grade conversational application designed to replicate the exact 
speed, aesthetics, and functionality of modern conversational AI using:
- LLM / Inference: Microsoft Azure AI Foundry (gpt-5.4-nano / custom deployments)
- Chat History Persistence: Microsoft Azure Blob Storage (tatvikbot container)
- User Interface: Streamlit with custom Tatvikgpt dark mode styling

Key Features & Optimizations:
1. Native 60fps Streaming:
   - Uses `st.write_stream` to stream tokens directly over WebSockets,
     eliminating React full-DOM rerenders and matching Tatvikgpt's ultra-fast response speed.
2. Non-Blocking Cloud Persistence:
   - Saves conversations to local disk instantly and syncs to Azure Blob Storage
     in asynchronous background threads without blocking the user interface.
3. Chronological Chat History (LIFO):
   - Grouped into 'Today', 'Yesterday', 'Previous 7 Days', 'Previous 30 Days'.
   - Max 15 conversations displayed by default with instant deep search.
===============================================================================
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime
from typing import Generator, Tuple, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. SDK Imports & Verification
# -----------------------------------------------------------------------------
# OpenAI Python SDK (>=1.30.0) is the official Microsoft-recommended client
# for connecting to Azure OpenAI and Azure AI Foundry project endpoints.
try:
    from openai import AzureOpenAI, OpenAI
    import openai
except ImportError:
    st.error("The `openai` package is missing. Please run `pip install -r requirements.txt`.")
    sys.exit(1)

# Import our pure Azure Blob storage manager
from storage import ChatStorage


# -----------------------------------------------------------------------------
# 2. Environment Configuration
# -----------------------------------------------------------------------------
# Load environment variables from .env file
load_dotenv(override=True)


def get_env_var(keys: list[str], default: str = "") -> str:
    """
    Helper function to inspect multiple environment variable aliases and return
    the first non-empty value found. Checks st.secrets (Streamlit Community Cloud)
    first, then falls back to os.environ (.env).
    """
    # 1. Check Streamlit Cloud Secrets (st.secrets)
    try:
        if hasattr(st, "secrets"):
            for key in keys:
                if key in st.secrets:
                    val = str(st.secrets[key]).strip()
                    if val:
                        return val
    except Exception:
        pass

    # 2. Check local environment (.env)
    for key in keys:
        val = os.getenv(key, "").strip()
        if val:
            return val
    return default


def load_config() -> dict[str, Any]:
    """
    Loads and normalizes all Azure AI Foundry settings from the environment or secrets.
    """
    api_key = get_env_var(["AZURE_AI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_API_KEY"])
    endpoint = get_env_var(["AZURE_AI_ENDPOINT", "AZURE_OPENAI_ENDPOINT", "AZURE_ENDPOINT"])
    deployment = get_env_var([
        "AZURE_AI_DEPLOYMENT_NAME",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
        "AZURE_DEPLOYMENT_NAME",
        "AZURE_MODEL_NAME",
    ], default="gpt-5.4-nano")
    api_version = get_env_var(
        ["AZURE_AI_API_VERSION", "AZURE_OPENAI_API_VERSION"],
        default="2024-08-01-preview",
    )
    endpoint_type = get_env_var(["AZURE_AI_ENDPOINT_TYPE"], default="auto").lower()

    system_prompt = get_env_var(
        ["SYSTEM_PROMPT"],
        default="You are an intelligent, helpful, and concise AI assistant powered by Azure AI Foundry.",
    )

    try:
        default_temp = float(get_env_var(["DEFAULT_TEMPERATURE"], "0.7"))
    except ValueError:
        default_temp = 0.7

    try:
        default_max_tokens = int(get_env_var(["DEFAULT_MAX_TOKENS"], "2048"))
    except ValueError:
        default_max_tokens = 2048

    return {
        "api_key": api_key,
        "endpoint": endpoint,
        "deployment": deployment,
        "api_version": api_version,
        "endpoint_type": endpoint_type,
        "system_prompt": system_prompt,
        "temperature": default_temp,
        "max_tokens": default_max_tokens,
    }


# -----------------------------------------------------------------------------
# 3. Streamlit Page Configuration & Authentic Tatvikgpt Dark Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tatvikgpt",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Azure Storage Singleton
storage = ChatStorage()
storage_status = storage.get_status()
env_config = load_config()

# Inject authentic Tatvikgpt CSS styling
st.markdown(
    """
    <style>
    /* Global Tatvikgpt dark mode palette: #212121 main background, #ececec text */
    .stApp {
        background-color: #212121;
        color: #ececec;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Center conversational container constrained to standard 768px width */
    .main .block-container {
        max-width: 768px;
        padding-top: 1.2rem;
        padding-bottom: 7rem;
    }

    /* Sidebar drawer matching Tatvikgpt's #171717 background */
    [data-testid="stSidebar"] {
        background-color: #171717 !important;
        border-right: 1px solid #2f2f2f;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
        padding-bottom: 1.5rem;
    }

    /* Minimalist Topbar */
    .tatvikgpt-topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0 0.8rem 0;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .model-badge {
        font-size: 1.15rem;
        font-weight: 600;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .model-chip {
        font-size: 0.72rem;
        background: #2f2f2f;
        color: #10a37f;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
    }
    
    /* Chronological Time Headers in Sidebar (Today, Yesterday, etc.) */
    .time-header {
        font-size: 0.72rem;
        font-weight: 600;
        color: #8e8e8e;
        text-transform: capitalize;
        margin: 1.1rem 0 0.35rem 0.4rem;
        letter-spacing: 0.02em;
    }

    /* Tatvikgpt Welcome Landing Screen */
    .welcome-container {
        text-align: center;
        padding: 4.5rem 1rem 2.5rem 1rem;
    }
    .welcome-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 2rem;
        letter-spacing: -0.03em;
    }

    /* Message rows without extra borders */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        padding: 1rem 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    }
    
    /* Floating Chat Input Pill (matching Tatvikgpt bottom bar) */
    [data-testid="stChatInput"] {
        border-radius: 26px !important;
        background-color: #2f2f2f !important;
        border: 1px solid #383838 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    [data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #ececec !important;
    }

    /* Footer Disclaimer */
    .tatvikgpt-footer {
        text-align: center;
        font-size: 0.72rem;
        color: #8e8e8e;
        margin-top: 1.5rem;
    }

    /* Telemetry metadata */
    .msg-meta {
        font-size: 0.72rem;
        color: #71717a;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 4. Azure Client Factory & Endpoint Resolver
# -----------------------------------------------------------------------------
def resolve_client(
    api_key: str,
    endpoint: str,
    api_version: str,
    endpoint_type_mode: str,
) -> Tuple[Optional[Any], str, Optional[str]]:
    """
    Initializes the appropriate OpenAI client instance for Azure AI Foundry.
    Handles:
    - Azure AI Foundry Project endpoints (routes via .../openai/v1)
    - Azure AI Foundry Serverless MaaS endpoints (routes via .../models)
    - Classic Azure OpenAI Service endpoints (*.openai.azure.com)
    """
    if not api_key:
        return None, "none", "API key missing in `.env` (`AZURE_AI_API_KEY`)."
    if not endpoint:
        return None, "none", "Endpoint URL missing in `.env` (`AZURE_AI_ENDPOINT`)."

    clean_endpoint = endpoint.strip().rstrip("/")
    is_project_endpoint = "/api/projects" in clean_endpoint
    is_serverless = (
        endpoint_type_mode == "foundry_serverless"
        or "services.ai.azure.com" in clean_endpoint
        or "inference.ai.azure.com" in clean_endpoint
        or clean_endpoint.endswith("/models")
    )

    try:
        if is_project_endpoint:
            # Azure AI Foundry Project endpoint:
            # When copied from portal, URLs often contain '/openai/v1/responses'.
            # The OpenAI SDK base_url must point to '.../openai/v1'
            base_url = clean_endpoint.split("/openai")[0] + "/openai/v1" if "/openai" in clean_endpoint else f"{clean_endpoint}/openai/v1"
            client = OpenAI(base_url=base_url, api_key=api_key)
            return client, "Foundry Project", None

        elif is_serverless and endpoint_type_mode != "azure_openai":
            # Serverless Models-as-a-Service (MaaS) endpoint
            base_url = clean_endpoint
            if not base_url.endswith("/models") and "services.ai.azure.com" in base_url:
                base_url = f"{base_url}/models"
            client = OpenAI(base_url=base_url, api_key=api_key)
            return client, "Foundry Serverless", None

        else:
            # Standard Azure OpenAI Service endpoint (*.openai.azure.com)
            if "/openai" in clean_endpoint:
                clean_endpoint = clean_endpoint.split("/openai")[0]
            client = AzureOpenAI(azure_endpoint=clean_endpoint, api_key=api_key, api_version=api_version)
            return client, "Azure OpenAI", None

    except Exception as e:
        return None, "error", str(e)


# -----------------------------------------------------------------------------
# 5. Session State Initialization
# -----------------------------------------------------------------------------
# Preserve active session across Streamlit reruns
if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"chat_{int(time.time())}_{uuid.uuid4().hex[:6]}"

if "session_title" not in st.session_state:
    st.session_state["session_title"] = "New chat"

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "selected_deployment" not in st.session_state:
    st.session_state["selected_deployment"] = env_config["deployment"]

active_deployment = st.session_state["selected_deployment"]

# Resolve Azure AI client
client, detected_mode, client_err = resolve_client(
    api_key=env_config["api_key"],
    endpoint=env_config["endpoint"],
    api_version=env_config["api_version"],
    endpoint_type_mode=env_config["endpoint_type"],
)
is_configured = bool(client and not client_err and active_deployment)


# -----------------------------------------------------------------------------
# 6. Sidebar: Tatvikgpt Navigation Drawer
# -----------------------------------------------------------------------------
with st.sidebar:
    # -------------------------------------------------------------------------
    # A. ➕ New Chat Button
    # -------------------------------------------------------------------------
    if st.button("➕ New chat", use_container_width=True, type="primary"):
        # Create fresh session ID and reset message thread
        st.session_state["session_id"] = f"chat_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        st.session_state["session_title"] = "New chat"
        st.session_state["messages"] = []
        st.rerun()

    # -------------------------------------------------------------------------
    # B. 🔍 Real-Time Instant Search Bar
    # -------------------------------------------------------------------------
    search_query = st.text_input(
        "Search chats",
        value="",
        placeholder="🔍 Search conversations...",
        label_visibility="collapsed",
        key="tatvikgpt_search_input",
    ).strip()

    # -------------------------------------------------------------------------
    # C. LIFO Conversation History (Max 15 Display Limit + Chronological Groups)
    # -------------------------------------------------------------------------
    all_chats = storage.list_chats()  # Instant sub-millisecond memory read
    total_count = len(all_chats)

    if not all_chats:
        st.caption("No conversations yet.")
    else:
        if search_query:
            # Search mode: queries all conversations by title, model, and message content
            filtered_chats = storage.search_chats(search_query)
            st.caption(f"Found {len(filtered_chats)} result(s)")
            grouped = {"Search Results": filtered_chats}
        else:
            # LIFO Mode: display only top 15 most recent chats
            recent_15 = all_chats[:15]
            # Group into Tatvikgpt chronological sections: Today, Yesterday, Previous 7 Days, etc.
            grouped = ChatStorage.group_chats_by_time(recent_15)

        # Render conversation items in a clean scrollable container
        history_container = st.container(height=320)
        with history_container:
            for group_label, chats_in_group in grouped.items():
                st.markdown(f'<div class="time-header">{group_label}</div>', unsafe_allow_html=True)
                for c in chats_in_group:
                    cid = c["session_id"]
                    ctitle = c.get("title", "New chat")
                    is_active = (cid == st.session_state["session_id"])

                    col_link, col_del = st.columns([0.84, 0.16])
                    with col_link:
                        btn_prefix = "💬 " if not is_active else "📍 "
                        if st.button(
                            f"{btn_prefix}{ctitle}",
                            key=f"item_{cid}",
                            use_container_width=True,
                            type="primary" if is_active else "secondary",
                        ):
                            # Load selected conversation into session state
                            loaded = storage.load_chat(cid)
                            if loaded:
                                st.session_state["session_id"] = cid
                                st.session_state["session_title"] = loaded.get("title", ctitle)
                                st.session_state["messages"] = loaded.get("messages", [])
                                st.rerun()

                    with col_del:
                        if st.button("🗑️", key=f"del_{cid}", help="Delete chat", use_container_width=True):
                            storage.delete_chat(cid)
                            if is_active:
                                st.session_state["session_id"] = f"chat_{int(time.time())}_{uuid.uuid4().hex[:6]}"
                                st.session_state["session_title"] = "New chat"
                                st.session_state["messages"] = []
                            st.rerun()

        # If there are conversations older than the top 15, allow expanding them
        if total_count > 15 and not search_query:
            older_chats = all_chats[15:]
            with st.expander(f"Older chats ({len(older_chats)})", expanded=False):
                for oc in older_chats:
                    ocid = oc["session_id"]
                    octitle = oc.get("title", "New chat")
                    if st.button(f"💬 {octitle}", key=f"old_{ocid}", use_container_width=True):
                        d = storage.load_chat(ocid)
                        if d:
                            st.session_state["session_id"] = ocid
                            st.session_state["session_title"] = d.get("title", octitle)
                            st.session_state["messages"] = d.get("messages", [])
                            st.rerun()

    st.divider()

    # -------------------------------------------------------------------------
    # D. Azure Storage Status
    # -------------------------------------------------------------------------
    storage_label = storage_status.get("label", "Local Cache")
    st.markdown(
        f"""
        <div style="font-size: 0.76rem; color: #8e8e8e; padding: 4px 6px;">
            Storage: <b style="color: #10a37f;">{storage_label}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # E. Model Deployment Selector
    # -------------------------------------------------------------------------
    known_models = ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-6-luna", "Custom..."]
    default_m_idx = known_models.index(active_deployment) if active_deployment in known_models[:3] else 3
    selected_m = st.selectbox(
        "Model",
        options=known_models,
        index=default_m_idx,
        label_visibility="collapsed",
    )
    if selected_m == "Custom...":
        custom_input = st.text_input("Custom Model", value=active_deployment)
        if custom_input:
            st.session_state["selected_deployment"] = custom_input.strip()
    else:
        st.session_state["selected_deployment"] = selected_m


    # Clear current chat
    if st.button("🗑️ Clear current chat", use_container_width=True):
        st.session_state["messages"] = []
        storage.delete_chat(st.session_state["session_id"])
        st.rerun()


# -----------------------------------------------------------------------------
# 7. Main Topbar
# -----------------------------------------------------------------------------
col_top_l, col_top_r = st.columns([0.75, 0.25])
with col_top_l:
    st.markdown(
        f"""
        <div class="tatvikgpt-topbar">
            <div class="model-badge">
                <span>Tatvikgpt</span>
                <span class="model-chip">{active_deployment}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_top_r:
    if st.session_state["messages"]:
        chat_json = json.dumps(
            {
                "session_id": st.session_state["session_id"],
                "title": st.session_state["session_title"],
                "model": active_deployment,
                "messages": st.session_state["messages"],
            },
            indent=2,
        )
        st.download_button(
            label="📥 Export",
            data=chat_json,
            file_name=f"{st.session_state['session_id']}.json",
            mime="application/json",
        )


# -----------------------------------------------------------------------------
# 8. Empty State: Tatvikgpt "What can I help with?"
# -----------------------------------------------------------------------------
if not st.session_state["messages"]:
    st.markdown(
        """
        <div class="welcome-container">
            <div class="welcome-title">What can I help with?</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    s_col1, s_col2 = st.columns(2)
    suggestions = [
        ("💡 Brainstorm ideas", "Brainstorm innovative enterprise AI use cases combining Azure AI Foundry and Blob Storage.", s_col1),
        ("💻 Write or debug code", "Write a production-ready Python FastAPI script that streams Azure AI chat completions.", s_col2),
        ("📊 Cloud Architecture", "Explain the difference between Azure AI Foundry serverless endpoints and Azure OpenAI resources.", s_col1),
        ("⚡ Summarize text", "Summarize the architectural benefits of using gpt-5.4-nano for high-frequency low-latency tasks.", s_col2),
    ]

    for title, prompt_text, col in suggestions:
        with col:
            if st.button(title, use_container_width=True, key=f"sugg_{title}"):
                now_t = datetime.now().strftime("%H:%M")
                st.session_state["session_title"] = ChatStorage.generate_title(prompt_text)
                st.session_state["messages"].append({
                    "role": "user",
                    "content": prompt_text,
                    "timestamp": now_t,
                })
                st.rerun()


# -----------------------------------------------------------------------------
# 9. Render Chat Message Thread
# -----------------------------------------------------------------------------
for msg in st.session_state["messages"]:
    role = msg["role"]
    content = msg["content"]
    timestamp = msg.get("timestamp", "")
    avatar = "👤" if role == "user" else "⚡"

    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        if timestamp and role == "assistant":
            st.markdown(f'<div class="msg-meta">{timestamp} • {active_deployment}</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 10. Native High-Speed WebSocket Token Streaming
# -----------------------------------------------------------------------------
def stream_azure_response(
    ai_client: Any,
    messages_payload: list[dict[str, str]],
    model_name: str,
    temp: float,
    max_toks: int,
) -> Generator[str, None, None]:
    """
    Yields completion delta chunks token-by-token from Azure AI Foundry.
    Handles 'max_completion_tokens' for modern GPT-5.x/o-series models.
    """
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages_payload,
        "stream": True,
        "temperature": temp,
    }

    try:
        response = ai_client.chat.completions.create(
            **kwargs,
            max_completion_tokens=max_toks,
        )
    except openai.BadRequestError as err:
        err_str = str(err).lower()
        if "max_completion_tokens" in err_str:
            # Fallback to legacy max_tokens for older model variants
            response = ai_client.chat.completions.create(
                **kwargs,
                max_tokens=max_toks,
            )
        elif "temperature" in err_str:
            # Handle reasoning models that only accept default temperature
            kwargs.pop("temperature", None)
            response = ai_client.chat.completions.create(
                **kwargs,
                max_completion_tokens=max_toks,
            )
        else:
            raise

    # Yield tokens as they stream in from Azure
    for chunk in response:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            delta_content = getattr(delta, "content", None)
            if delta_content:
                yield delta_content


# -----------------------------------------------------------------------------
# 11. Floating Chat Input & Response Execution
# -----------------------------------------------------------------------------
user_input = st.chat_input(
    placeholder="Message Tatvikgpt..." if is_configured else "Please configure .env to start...",
    disabled=not is_configured,
)

if user_input:
    # Auto-generate dynamic title on first message
    if st.session_state["session_title"] == "New chat":
        st.session_state["session_title"] = ChatStorage.generate_title(user_input)

    # 1. Immediately append and display the user's message
    now_t = datetime.now().strftime("%H:%M")
    st.session_state["messages"].append({
        "role": "user",
        "content": user_input,
        "timestamp": now_t,
    })

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # 2. Build conversational context payload
    api_messages = [{"role": "system", "content": env_config["system_prompt"]}]
    for m in st.session_state["messages"]:
        api_messages.append({"role": m["role"], "content": m["content"]})

    # 3. Stream assistant response at 60fps using Streamlit native st.write_stream
    with st.chat_message("assistant", avatar="⚡"):
        start_t = time.time()
        try:
            stream = stream_azure_response(
                ai_client=client,
                messages_payload=api_messages,
                model_name=active_deployment,
                temp=env_config["temperature"],
                max_toks=env_config["max_tokens"],
            )

            # st.write_stream streams directly over WebSockets with 0 DOM re-render lag
            full_response = st.write_stream(stream)
            elapsed = time.time() - start_t

            # Display response telemetry
            st.markdown(f'<div class="msg-meta">{now_t} • {active_deployment} • ⚡ {elapsed:.2f}s</div>', unsafe_allow_html=True)

            # Record completed response in session state
            st.session_state["messages"].append({
                "role": "assistant",
                "content": full_response,
                "timestamp": now_t,
            })

            # Auto-save conversation to Azure Blob Storage in background thread (0ms delay)
            storage.save_chat(
                session_id=st.session_state["session_id"],
                title=st.session_state["session_title"],
                messages=st.session_state["messages"],
                model=active_deployment,
            )

        except Exception as e:
            st.error(f"Error communicating with Azure AI Foundry: {e}")

# Subtle disclaimer footer
st.markdown('<div class="tatvikgpt-footer">Tatvikgpt can make mistakes. Check important info.</div>', unsafe_allow_html=True)
