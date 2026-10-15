"""
===============================================================================
AZURE BLOB CHAT STORAGE MODULE (Pure Azure Architecture)
===============================================================================
This module manages conversational history persistence and file/blob attachments
using Microsoft Azure Blob Storage (azure-storage-blob SDK) with an automatic local 
disk cache fallback.

Key Architectural Decisions:
1. Pure Azure Ecosystem:
   - Dedicated solely to Azure Blob Storage; all AWS/S3 dependencies removed.
2. Zero-Latency User Experience:
   - Synchronous writes to local cache guarantee <1ms read/write response.
   - Asynchronous background threads (threading.Thread) upload JSON blobs to Azure 
     so the Streamlit UI is never blocked by cloud network latency.
3. In-Memory Caching:
   - Loads chat metadata into an in-memory dictionary on initialization, 
     delivering instant (0ms) conversation listing and switching.
4. Tatvikgpt-Style Organization:
   - Chronological grouping: "Today", "Yesterday", "Previous 7 Days", "Older".
   - Strict LIFO (Last-In-First-Out) sorting with a 15-conversation display window.
===============================================================================
"""

import os
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv

# Import the official Azure Storage Blob SDK
try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
    from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
    AZURE_BLOB_AVAILABLE = True
except ImportError:
    AZURE_BLOB_AVAILABLE = False

# Load environment configuration from .env file
load_dotenv(override=True)

logger = logging.getLogger("AzureBlobChatStorage")

# Local cache directories for offline capability and instant sub-millisecond reads
LOCAL_CACHE_DIR = Path(__file__).parent / ".chat_history"
LOCAL_CACHE_DIR.mkdir(exist_ok=True)

LOCAL_ATTACHMENTS_DIR = LOCAL_CACHE_DIR / "attachments"
LOCAL_ATTACHMENTS_DIR.mkdir(exist_ok=True)


def get_storage_setting(keys: List[str], default: str = "") -> str:
    """
    Retrieves storage settings from st.secrets (Streamlit Community Cloud)
    first, falling back to os.environ (.env).
    """
    # 1. Check Streamlit Cloud Secrets (st.secrets)
    try:
        import streamlit as st
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


class ChatStorage:
    """
    Manages conversational history and blob attachments directly in Azure Blob Storage.
    Implements a Singleton pattern to reuse connection pools across Streamlit reruns.
    """

    _instance = None

    def __new__(cls):
        # Singleton pattern prevents reconnecting and re-authenticating on every page rerun
        if cls._instance is None:
            cls._instance = super(ChatStorage, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # Guard against re-initializing an already active singleton
        if getattr(self, "_initialized", False):
            return

        # ---------------------------------------------------------------------
        # 1. Read Azure Storage Configuration from Environment or Secrets
        # ---------------------------------------------------------------------
        # Option A: Full Connection String (Recommended)
        self.azure_conn_str = get_storage_setting([
            "AZURE_STORAGE_CONNECTION_STRING",
            "AZURE_STORAGE_CONN_STR",
        ])

        # Option B: Storage Account Name + Access Key (Alternative)
        self.azure_account_name = get_storage_setting(["AZURE_STORAGE_ACCOUNT_NAME"])
        self.azure_account_key = get_storage_setting(["AZURE_STORAGE_ACCOUNT_KEY"])

        # Target container name (default: 'chat-history')
        self.azure_container_name = get_storage_setting(
            ["AZURE_STORAGE_CONTAINER_NAME"],
            default="chat-history",
        )

        # Blob folder prefix (virtual folder inside container)
        self.azure_prefix = get_storage_setting(
            ["AZURE_STORAGE_BLOB_PREFIX"],
            default="chat_history/",
        )
        if not self.azure_prefix.endswith("/"):
            self.azure_prefix += "/"

        # Internal state
        self.is_azure_connected = False
        self.azure_blob_service = None
        self.azure_container_client = None

        # In-memory dictionary storing conversation metadata for 0ms reads
        # Schema: { session_id: { "session_id", "title", "updated_at", "model", "message_count" } }
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._initial_cloud_synced = False

        # ---------------------------------------------------------------------
        # 2. Initialize Azure Backend & Local Cache
        # ---------------------------------------------------------------------
        self._init_azure_client()
        self._load_local_disk_cache()

        # Launch background thread to sync existing cloud blobs into local cache
        if self.is_azure_connected:
            threading.Thread(target=self._sync_blobs_from_azure, daemon=True).start()

        self._initialized = True

    def _init_azure_client(self):
        """
        Initializes the BlobServiceClient and gets/creates the destination container.
        """
        if not AZURE_BLOB_AVAILABLE:
            logger.warning("azure-storage-blob library is not installed. Using local cache.")
            self.is_azure_connected = False
            return

        # Check if credentials are present
        if not (self.azure_conn_str or (self.azure_account_name and self.azure_account_key)):
            logger.info("No Azure Storage credentials found in .env. Using local directory cache.")
            self.is_azure_connected = False
            return

        try:
            # Connect via Connection String or Account Key
            if self.azure_conn_str:
                self.azure_blob_service = BlobServiceClient.from_connection_string(self.azure_conn_str)
            else:
                account_url = f"https://{self.azure_account_name}.blob.core.windows.net"
                self.azure_blob_service = BlobServiceClient(account_url=account_url, credential=self.azure_account_key)

            # Get ContainerClient for the designated container
            self.azure_container_client = self.azure_blob_service.get_container_client(self.azure_container_name)

            # Auto-create container if it does not already exist
            try:
                self.azure_container_client.create_container()
                logger.info(f"Created Azure Blob container: '{self.azure_container_name}'")
            except ResourceExistsError:
                # Container already exists, which is normal
                pass

            self.is_azure_connected = True
            logger.info(f"Successfully connected to Azure Blob Storage container: '{self.azure_container_name}'")

        except Exception as e:
            logger.error(f"Failed to connect to Azure Blob Storage: {e}. Falling back to local cache.")
            self.is_azure_connected = False

    def _load_local_disk_cache(self):
        """
        Reads local JSON files into the in-memory cache on startup.
        Ensures instantaneous sidebar rendering without touching the network.
        """
        for file_path in LOCAL_CACHE_DIR.glob("*.json"):
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                sid = data.get("session_id", file_path.stem)
                self._memory_cache[sid] = {
                    "session_id": sid,
                    "title": data.get("title", "New chat"),
                    "updated_at": data.get("updated_at", ""),
                    "model": data.get("model", ""),
                    "message_count": len(data.get("messages", [])),
                }
            except Exception:
                continue

    def _sync_blobs_from_azure(self):
        """
        Runs once in a background thread on startup to pull down any chat history
        blobs stored in Azure that might not yet exist on this local machine.
        """
        if self._initial_cloud_synced or not self.is_azure_connected:
            return
        self._initial_cloud_synced = True

        try:
            # List all blobs under our designated prefix
            blob_list = self.azure_container_client.list_blobs(name_starts_with=self.azure_prefix)
            for blob in blob_list:
                if not blob.name.endswith(".json"):
                    continue

                session_id = Path(blob.name).stem
                local_file = self._local_path(session_id)

                # Download only if file is not already in local cache
                if not local_file.exists():
                    try:
                        blob_client = self.azure_container_client.get_blob_client(blob.name)
                        blob_content = blob_client.download_blob().readall().decode("utf-8")
                        local_file.write_text(blob_content, encoding="utf-8")

                        parsed_data = json.loads(blob_content)
                        self._memory_cache[session_id] = {
                            "session_id": session_id,
                            "title": parsed_data.get("title", "New chat"),
                            "updated_at": parsed_data.get("updated_at", blob.last_modified.isoformat() if blob.last_modified else ""),
                            "model": parsed_data.get("model", ""),
                            "message_count": len(parsed_data.get("messages", [])),
                        }
                    except Exception as err:
                        logger.warning(f"Error downloading blob {blob.name}: {err}")

        except Exception as e:
            logger.warning(f"Initial Azure Blob sync failed: {e}")

    def get_status(self) -> Dict[str, Any]:
        """
        Returns connection status for UI display in the sidebar.
        """
        if self.is_azure_connected:
            return {
                "is_azure": True,
                "label": f"Azure Blob ({self.azure_container_name})",
                "container": self.azure_container_name,
                "badge_class": "badge-ready",
            }
        return {
            "is_azure": False,
            "label": "Local Storage (.chat_history)",
            "container": "Local Cache",
            "badge_class": "badge-warning",
        }

    def _azure_blob_name(self, session_id: str) -> str:
        """Constructs the full virtual path for the JSON blob inside the Azure container."""
        return f"{self.azure_prefix}{session_id}.json"

    def _local_path(self, session_id: str) -> Path:
        """Constructs the local disk cache path for a given session ID."""
        return LOCAL_CACHE_DIR / f"{session_id}.json"

    def save_chat(
        self,
        session_id: str,
        title: str,
        messages: List[Dict[str, Any]],
        model: str = "",
    ) -> bool:
        """
        Saves the conversation to local disk instantly, then triggers a background thread
        to upload the JSON blob to Azure Blob Storage without blocking the UI.
        """
        if not session_id or not messages:
            return False

        updated_at = datetime.now().isoformat()
        payload = {
            "session_id": session_id,
            "title": title or "New chat",
            "model": model,
            "updated_at": updated_at,
            "messages": messages,
        }
        json_data = json.dumps(payload, ensure_ascii=False, indent=2)

        # 1. Update in-memory cache for instant UI rendering (0ms latency)
        self._memory_cache[session_id] = {
            "session_id": session_id,
            "title": title or "New chat",
            "updated_at": updated_at,
            "model": model,
            "message_count": len(messages),
        }

        # 2. Write to local file (<1ms latency)
        try:
            self._local_path(session_id).write_text(json_data, encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed writing local cache: {e}")

        # 3. Asynchronous cloud upload to Azure Blob Storage
        # Running in a daemon thread ensures the user's chat stream finishes instantly
        # without waiting for an HTTPS round-trip to Azure.
        if self.is_azure_connected:
            threading.Thread(
                target=self._upload_blob_async,
                args=(session_id, json_data),
                daemon=True,
            ).start()

        return True

    def _upload_blob_async(self, session_id: str, json_data: str):
        """
        Helper executed inside a background thread to upload JSON to Azure Blob Storage.
        """
        if not self.is_azure_connected or not self.azure_container_client:
            return

        try:
            blob_name = self._azure_blob_name(session_id)
            blob_client = self.azure_container_client.get_blob_client(blob_name)
            content_settings = ContentSettings(content_type="application/json") if ContentSettings else None
            
            # upload_blob with overwrite=True replaces previous turns with the updated thread
            blob_client.upload_blob(
                json_data.encode("utf-8"),
                overwrite=True,
                content_settings=content_settings,
            )
        except Exception as e:
            logger.error(f"Background Azure Blob upload failed for {session_id}: {e}")

    def load_chat(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a full conversation history (messages, title, timestamp).
        Reads from local cache (<1ms) or downloads from Azure Blob if missing locally.
        """
        if not session_id:
            return None

        # 1. Check local cache (fastest)
        local_file = self._local_path(session_id)
        if local_file.exists():
            try:
                return json.loads(local_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        # 2. Fallback to Azure Blob Storage
        if self.is_azure_connected and self.azure_container_client:
            try:
                blob_name = self._azure_blob_name(session_id)
                blob_client = self.azure_container_client.get_blob_client(blob_name)
                raw_json = blob_client.download_blob().readall().decode("utf-8")
                # Cache locally for subsequent fast reads
                local_file.write_text(raw_json, encoding="utf-8")
                return json.loads(raw_json)
            except Exception:
                pass

        return None

    def list_chats(self) -> List[Dict[str, Any]]:
        """
        Returns all conversations sorted in LIFO order (Last-In-First-Out = most recent first).
        Reads directly from in-memory cache to guarantee instantaneous UI navigation.
        """
        return sorted(
            self._memory_cache.values(),
            key=lambda item: item.get("updated_at", ""),
            reverse=True,
        )

    def search_chats(self, query: str = "") -> List[Dict[str, Any]]:
        """
        Performs real-time search across ALL conversations.
        Matches against conversation titles, models, and message contents.
        """
        all_chats = self.list_chats()
        if not query or not query.strip():
            return all_chats

        q = query.strip().lower()
        matching_results = []

        for chat in all_chats:
            sid = chat["session_id"]
            title = chat.get("title", "").lower()
            model = chat.get("model", "").lower()

            # Fast match on title or model name
            if q in title or q in model:
                matching_results.append(chat)
                continue

            # Deep search inside message content
            chat_data = self.load_chat(sid)
            if chat_data:
                for msg in chat_data.get("messages", []):
                    if q in msg.get("content", "").lower():
                        matching_results.append(chat)
                        break

        return matching_results

    def delete_chat(self, session_id: str) -> bool:
        """
        Deletes a conversation from memory and local disk immediately,
        then permanently removes the blob from Azure Storage asynchronously.
        """
        # Remove from in-memory cache
        self._memory_cache.pop(session_id, None)

        # Remove local file
        local_file = self._local_path(session_id)
        if local_file.exists():
            try:
                local_file.unlink()
            except Exception:
                pass

        # Delete blob from Azure in background thread
        if self.is_azure_connected:
            threading.Thread(
                target=self._delete_blob_async,
                args=(session_id,),
                daemon=True,
            ).start()

        return True

    def _delete_blob_async(self, session_id: str):
        """Helper executed in background thread to delete blob from Azure Storage."""
        if not self.is_azure_connected or not self.azure_container_client:
            return

        try:
            blob_name = self._azure_blob_name(session_id)
            blob_client = self.azure_container_client.get_blob_client(blob_name)
            blob_client.delete_blob()
        except Exception as e:
            logger.error(f"Failed to delete blob {session_id} from Azure: {e}")

    def upload_attachment_blob(
        self,
        file_name: str,
        file_bytes: bytes,
        content_type: str = "application/octet-stream",
    ) -> Tuple[bool, str]:
        """
        Uploads an attachment file directly into Azure Blob Storage under 'attachments/<filename>'.
        Returns: (success: bool, url_or_path: str)
        """
        clean_name = Path(file_name).name
        timestamp_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_name = f"{timestamp_prefix}_{clean_name}"
        local_target = LOCAL_ATTACHMENTS_DIR / unique_name

        # Always save a local copy
        try:
            local_target.write_bytes(file_bytes)
        except Exception as e:
            logger.error(f"Failed writing local attachment: {e}")

        # Upload to Azure Blob Storage container
        if self.is_azure_connected and self.azure_container_client:
            try:
                blob_name = f"attachments/{unique_name}"
                blob_client = self.azure_container_client.get_blob_client(blob_name)
                content_settings = ContentSettings(content_type=content_type) if ContentSettings else None
                blob_client.upload_blob(file_bytes, overwrite=True, content_settings=content_settings)
                return True, blob_client.url
            except Exception as e:
                logger.error(f"Azure attachment upload failed: {e}")
                return False, str(local_target)

        return True, str(local_target)

    @staticmethod
    def group_chats_by_time(chats: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Groups conversations into Tatvikgpt chronological buckets:
        'Today', 'Yesterday', 'Previous 7 Days', 'Previous 30 Days', 'Older'.
        """
        now = datetime.now()
        today = now.date()
        yesterday = today - timedelta(days=1)
        seven_days_ago = today - timedelta(days=7)
        thirty_days_ago = today - timedelta(days=30)

        groups: Dict[str, List[Dict[str, Any]]] = {
            "Today": [],
            "Yesterday": [],
            "Previous 7 Days": [],
            "Previous 30 Days": [],
            "Older": [],
        }

        for chat in chats:
            updated_str = chat.get("updated_at", "")
            try:
                chat_date = datetime.fromisoformat(updated_str).date()
                if chat_date == today:
                    groups["Today"].append(chat)
                elif chat_date == yesterday:
                    groups["Yesterday"].append(chat)
                elif chat_date > seven_days_ago:
                    groups["Previous 7 Days"].append(chat)
                elif chat_date > thirty_days_ago:
                    groups["Previous 30 Days"].append(chat)
                else:
                    groups["Older"].append(chat)
            except Exception:
                groups["Older"].append(chat)

        return {k: v for k, v in groups.items() if v}

    @staticmethod
    def generate_title(first_prompt: str) -> str:
        """
        Auto-generates a clean, concise title from the user's initial prompt (just like Tatvikgpt).
        """
        cleaned = first_prompt.strip().replace("\n", " ")
        if len(cleaned) <= 32:
            return cleaned
        return cleaned[:30].rstrip() + "..."
