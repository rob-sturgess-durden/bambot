import os
import re
import json
import logging
import base64
from datetime import datetime
from functools import lru_cache
from threading import Lock
from typing import Dict, Optional, List, Any, Tuple

from training_curriculum import build_training_context, get_total_steps, get_topic_info, is_topic_complete

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response, StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field

try:  # Optional dependency until authentication is configured
    import bcrypt  # type: ignore
except Exception:  # pragma: no cover
    bcrypt = None

try:  # Optional dependency until Google Sheets logging is configured
    import gspread  # type: ignore
    from gspread.utils import rowcol_to_a1  # type: ignore
    from google.oauth2.service_account import Credentials  # type: ignore
    from google.auth.exceptions import GoogleAuthError  # type: ignore
except Exception:  # pragma: no cover
    gspread = None  # type: ignore
    Credentials = None  # type: ignore
    rowcol_to_a1 = None  # type: ignore
    GoogleAuthError = Exception

try:
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    import multipart  # type: ignore  # noqa: F401
    HAS_MULTIPART = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    HAS_MULTIPART = False

DEFAULT_PROMPT_ID = "pmpt_68dafc0b9a408195836b76108153ee8e0fce762e4ce36e20"
DEFAULT_VECTOR_STORE_ID = "vs_68dafbe90f3c81919d396ebafc21031d,vs_68dae6d13bd4819194d81215bd0c3874"

PROMPT_ID = os.getenv("OPENAI_PROMPT_ID", os.getenv("PROMPT_ID") or DEFAULT_PROMPT_ID)
PROMPT_VERSION = os.getenv("OPENAI_PROMPT_VERSION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RESPONSES_MODEL = os.getenv("OPENAI_RESPONSES_MODEL")
VECTOR_STORE_IDS = [
    item.strip()
    for item in os.getenv("OPENAI_VECTOR_STORE_IDS", DEFAULT_VECTOR_STORE_ID).split(",")
    if item.strip()
]
RESPONSES_INCLUDE_FIELDS = [
    item.strip()
    for item in os.getenv(
        "OPENAI_RESPONSES_INCLUDE",
        "reasoning.encrypted_content,web_search_call.action.sources",
    ).split(",")
    if item.strip()
]
STORE_RESPONSES = os.getenv("OPENAI_STORE_RESPONSES", "true").lower() in {"1", "true", "yes", "on"}

CONTROLLER_PROMPT_ID = os.getenv("OPENAI_CONTROLLER_PROMPT_ID", PROMPT_ID)
CONTROLLER_PROMPT_VERSION = os.getenv("OPENAI_CONTROLLER_PROMPT_VERSION")
TRAINER_PROMPT_ID = os.getenv("OPENAI_TRAINER_PROMPT_ID", PROMPT_ID)
TRAINER_PROMPT_VERSION = os.getenv("OPENAI_TRAINER_PROMPT_VERSION")
PRACTICER_PROMPT_ID = os.getenv("OPENAI_PRACTICER_PROMPT_ID", PROMPT_ID)
PRACTICER_PROMPT_VERSION = os.getenv("OPENAI_PRACTICER_PROMPT_VERSION")
COACH_PROMPT_ID = os.getenv("OPENAI_COACH_PROMPT_ID", PROMPT_ID)
COACH_PROMPT_VERSION = os.getenv("OPENAI_COACH_PROMPT_VERSION")
PRACTICER_REALTIME_PROMPT_ID = os.getenv("OPENAI_PRACTICER_REALTIME_PROMPT_ID")

AUTH_COOKIE_NAME = "bm_user"
AUTH_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 30

GOOGLE_SERVICE_ACCOUNT_INFO = os.getenv("GOOGLE_SERVICE_ACCOUNT_INFO") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
GOOGLE_USERS_WORKSHEET = os.getenv("GOOGLE_SHEET_USERS_TAB", "Users")
GOOGLE_INTERACTIONS_WORKSHEET = os.getenv("GOOGLE_SHEET_INTERACTIONS_TAB", "Interactions")
GOOGLE_SESSION_WORKSHEET = os.getenv("GOOGLE_SHEET_SESSION_STATE_TAB", "AgentState")
GOOGLE_PRACTICE_WORKSHEET = os.getenv("GOOGLE_SHEET_PRACTICE_TAB", "PracticeHistory")
GOOGLE_FEEDBACK_WORKSHEET = os.getenv("GOOGLE_SHEET_FEEDBACK_TAB", "CoachingFeedback")
GOOGLE_PROGRESS_WORKSHEET = os.getenv("GOOGLE_SHEET_PROGRESS_TAB", "Progress")
GOOGLE_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

USERS_HEADERS = ["timestamp", "email", "full_name", "password_hash"]
INTERACTION_HEADERS = ["timestamp", "user_id", "user_message", "assistant_reply"]
SESSION_STATE_HEADERS = ["user_id", "state_json", "updated_at"]
PRACTICE_HISTORY_HEADERS = ["timestamp", "user_id", "scenario", "difficulty", "transcript_json", "outcome"]
COACHING_FEEDBACK_HEADERS = ["timestamp", "user_id", "feedback", "next_steps", "rubric_json"]
PROGRESS_HEADERS = ["user_id", "completed_topics", "completed_practices", "completed_coaching_sessions", "last_activity", "last_task", "updated_at"]
TRAINING_DETAILS_HEADERS = ["timestamp", "user_id", "topic_id", "topic_title", "steps_completed"]

worksheet_lock = Lock()
logger = logging.getLogger(__name__)

app = FastAPI()

# Adjust CORS origins as needed (kept permissive)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    audio_mode: Optional[bool] = False


class ChatResponse(BaseModel):
    reply: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = ""


class LoginRequest(BaseModel):
    email: str
    password: str
    remember_me: Optional[bool] = False


class AuthResponse(BaseModel):
    email: str
    full_name: Optional[str] = ""


class AgentMessage(BaseModel):
    role: str
    content: str


class ControllerRequest(BaseModel):
    user_id: str
    message: str


class TrainerRequest(BaseModel):
    user_id: str
    message: str
    topic: Optional[str] = None
    mark_complete: Optional[bool] = False
    notes: Optional[str] = None


class PracticerRequest(BaseModel):
    user_id: str
    message: str
    scenario: str
    difficulty: Optional[str] = None
    outcome: Optional[str] = None
    transcript: List[AgentMessage] = Field(default_factory=list)
    session_completed: Optional[bool] = False


class CoachRequest(BaseModel):
    user_id: str
    message: str
    transcript: List[AgentMessage]
    previous_feedback: Optional[List[str]] = Field(default_factory=list)
    rubric: Optional[Dict[str, Any]] = None
    next_steps: Optional[str] = None


class AgentResponse(BaseModel):
    reply: str
    state: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


def _default_user_state() -> Dict[str, Any]:
    return {
        "current_role": "controller",
        "controller_history": [],
        "trainer_history": [],
        "practicer_history": [],
        "coach_history": [],
        "training_progress": {
            "completed_topics": [],
            "in_progress_topic": None,
            "current_step": 1,  # Current step within the topic
            "awaiting_response": False,
            "notes": "",
        },
        "practice_history": [],
        "coaching_feedback": [],
    }


def _sanitize_state_for_storage(state: Dict[str, Any]) -> Dict[str, Any]:
    default = _default_user_state()
    training = state.get("training_progress", {}) if isinstance(state, dict) else {}
    practice = state.get("practice_history", []) if isinstance(state, dict) else []
    coaching = state.get("coaching_feedback", []) if isinstance(state, dict) else []
    controller_history = state.get("controller_history", []) if isinstance(state, dict) else []
    trainer_history = state.get("trainer_history", []) if isinstance(state, dict) else []
    practicer_history = state.get("practicer_history", []) if isinstance(state, dict) else []
    coach_history = state.get("coach_history", []) if isinstance(state, dict) else []

    def _trim_history(items: List[Any], limit: int = 10) -> List[Any]:
        if not isinstance(items, list):
            return []
        return items[-limit:]

    sanitized = {
        "current_role": state.get("current_role", default["current_role"]),
        "controller_history": _trim_history(controller_history),
        "training_progress": {
            "completed_topics": training.get("completed_topics", [])[:20]
            if isinstance(training, dict)
            else [],
            "in_progress_topic": training.get("in_progress_topic") if isinstance(training, dict) else None,
            "current_step": int(training.get("current_step", 1)) if isinstance(training, dict) else 1,
            "awaiting_response": bool(training.get("awaiting_response", False)) if isinstance(training, dict) else False,
            "notes": str(training.get("notes", "")) if isinstance(training, dict) else "",
        },
        "practice_history": _trim_history(practice),
        "coaching_feedback": _trim_history(coaching),
        "trainer_history": _trim_history(trainer_history),
        "practicer_history": _trim_history(practicer_history),
        "coach_history": _trim_history(coach_history),
    }
    return sanitized


def _load_user_state(user_id: str) -> Tuple[Dict[str, Any], Optional[int]]:
    state = _default_user_state()
    if not user_id or not _google_sheets_available():
        return state, None
    try:
        worksheet = _ensure_worksheet(GOOGLE_SESSION_WORKSHEET, SESSION_STATE_HEADERS)
        records = worksheet.get_all_records(expected_headers=SESSION_STATE_HEADERS)
        for idx, row in enumerate(records, start=2):
            if str(row.get("user_id", "")).strip().lower() == user_id.strip().lower():
                raw_state = row.get("state_json", "{}")
                try:
                    loaded = json.loads(raw_state) if raw_state else {}
                except json.JSONDecodeError:
                    loaded = {}
                merged = _default_user_state()
                if isinstance(loaded, dict):
                    for key in merged.keys():
                        if key in loaded:
                            merged[key] = loaded[key]
                    # Manual deep merge for nested structures
                    training = loaded.get("training_progress", {}) if isinstance(loaded, dict) else {}
                    if isinstance(training, dict):
                        merged.setdefault("training_progress", {}).update(training)
                    for history_key in [
                        "controller_history",
                        "trainer_history",
                        "practicer_history",
                        "coach_history",
                        "practice_history",
                        "coaching_feedback",
                    ]:
                        if isinstance(loaded.get(history_key), list):
                            merged[history_key] = loaded[history_key]
                    if loaded.get("current_role"):
                        merged["current_role"] = loaded["current_role"]
                state = merged
                return state, idx
    except Exception as exc:  # pragma: no cover - network/API errors
        logger.warning("Failed to load session state: %s", exc)
    return state, None


def _save_user_state(user_id: str, state: Dict[str, Any], row_number: Optional[int] = None) -> None:
    if not user_id or not _google_sheets_available():
        return
    worksheet = _ensure_worksheet(GOOGLE_SESSION_WORKSHEET, SESSION_STATE_HEADERS)
    sanitized = _sanitize_state_for_storage(state)
    record = [
        user_id,
        json.dumps(sanitized, ensure_ascii=False),
        datetime.utcnow().isoformat(),
    ]
    with worksheet_lock:
        if row_number is None:
            # Attempt to find existing row before appending
            try:
                records = worksheet.get_all_records(expected_headers=SESSION_STATE_HEADERS)
                for idx, row in enumerate(records, start=2):
                    if str(row.get("user_id", "")).strip().lower() == user_id.strip().lower():
                        row_number = idx
                        break
            except Exception as exc:  # pragma: no cover - fallback to append
                logger.warning("Unable to locate existing session row: %s", exc)
        if row_number is not None:
            start_col = "A"
            end_col = chr(ord("A") + len(record) - 1)
            worksheet.update(f"{start_col}{row_number}:{end_col}{row_number}", [record])
        else:
            worksheet.append_row(record, value_input_option="USER_ENTERED")


def _append_practice_history(
    user_id: str,
    scenario: str,
    difficulty: Optional[str],
    transcript: List[Dict[str, str]],
    outcome: Optional[str],
) -> None:
    if not _google_sheets_available():
        return
    worksheet = _ensure_worksheet(GOOGLE_PRACTICE_WORKSHEET, PRACTICE_HISTORY_HEADERS)
    record = [
        datetime.utcnow().isoformat(),
        user_id,
        scenario,
        difficulty or "",
        json.dumps(transcript, ensure_ascii=False),
        outcome or "",
    ]
    with worksheet_lock:
        worksheet.append_row(record, value_input_option="USER_ENTERED")


def _append_coaching_feedback(
    user_id: str,
    feedback: str,
    next_steps: Optional[str],
    rubric: Optional[Dict[str, Any]],
) -> None:
    if not _google_sheets_available():
        return
    worksheet = _ensure_worksheet(GOOGLE_FEEDBACK_WORKSHEET, COACHING_FEEDBACK_HEADERS)
    record = [
        datetime.utcnow().isoformat(),
        user_id,
        feedback,
        next_steps or "",
        json.dumps(rubric or {}, ensure_ascii=False),
    ]
    with worksheet_lock:
        worksheet.append_row(record, value_input_option="USER_ENTERED")


def _append_training_completion(user_id: str, topic_id: str, steps_completed: int) -> None:
    if not _google_sheets_available():
        return

    worksheet = _ensure_worksheet("TrainingDetails", TRAINING_DETAILS_HEADERS)
    topic = get_topic_info(topic_id)
    record = [
        datetime.utcnow().isoformat(),
        user_id,
        topic_id,
        topic.get("title", "") if isinstance(topic, dict) else "",
        steps_completed,
    ]
    with worksheet_lock:
        worksheet.append_row(record, value_input_option="USER_ENTERED")


def _render_progress_bar(current_step: int, total_steps: int, awaiting_response: bool) -> str:
    if total_steps <= 0:
        return f"Step {current_step}"

    completed_steps = max(0, min(total_steps, current_step - 1))
    bar_length = 10
    filled_segments = min(bar_length, max(0, int((completed_steps / total_steps) * bar_length)))
    bar = "#" * filled_segments + "-" * (bar_length - filled_segments)
    status = "awaiting learner response" if awaiting_response else "introducing new step"
    return f"[{bar}] Step {current_step} of {total_steps} ({status})"


def _is_step_completed_reply(reply_text: str) -> bool:
    if not reply_text:
        return False
    return reply_text.lstrip().startswith("✅")


def _load_user_progress(user_id: str) -> Dict[str, Any]:
    """Load user progress from the Progress worksheet"""
    default_progress = {
        "completed_topics": [],
        "completed_practices": 0,
        "completed_coaching_sessions": 0,
        "last_activity": None,
        "last_task": None,
    }

    if not user_id or not _google_sheets_available():
        return default_progress

    try:
        worksheet = _ensure_worksheet(GOOGLE_PROGRESS_WORKSHEET, PROGRESS_HEADERS)
        records = worksheet.get_all_records(expected_headers=PROGRESS_HEADERS)
        for row in records:
            if str(row.get("user_id", "")).strip().lower() == user_id.strip().lower():
                completed_topics_str = row.get("completed_topics", "")
                completed_topics = completed_topics_str.split(",") if completed_topics_str else []
                completed_topics = [t.strip() for t in completed_topics if t.strip()]

                return {
                    "completed_topics": completed_topics,
                    "completed_practices": int(row.get("completed_practices", 0) or 0),
                    "completed_coaching_sessions": int(row.get("completed_coaching_sessions", 0) or 0),
                    "last_activity": row.get("last_activity") or None,
                    "last_task": row.get("last_task") or None,
                }
    except Exception as exc:
        logger.warning("Failed to load user progress: %s", exc)

    return default_progress


def _save_user_progress(user_id: str, progress: Dict[str, Any]) -> None:
    """Save user progress to the Progress worksheet"""
    if not user_id or not _google_sheets_available():
        return

    try:
        worksheet = _ensure_worksheet(GOOGLE_PROGRESS_WORKSHEET, PROGRESS_HEADERS)

        # Convert completed_topics list to comma-separated string
        completed_topics_str = ",".join(progress.get("completed_topics", []))

        record = [
            user_id,
            completed_topics_str,
            progress.get("completed_practices", 0),
            progress.get("completed_coaching_sessions", 0),
            progress.get("last_activity", ""),
            progress.get("last_task", ""),
            datetime.utcnow().isoformat(),
        ]

        # Find existing row or append new one
        with worksheet_lock:
            records = worksheet.get_all_records(expected_headers=PROGRESS_HEADERS)
            row_number = None
            for idx, row in enumerate(records, start=2):
                if str(row.get("user_id", "")).strip().lower() == user_id.strip().lower():
                    row_number = idx
                    break

            if row_number is not None:
                start_col = "A"
                end_col = chr(ord("A") + len(record) - 1)
                worksheet.update(f"{start_col}{row_number}:{end_col}{row_number}", [record])
            else:
                worksheet.append_row(record, value_input_option="USER_ENTERED")
    except Exception as exc:
        logger.warning("Failed to save user progress: %s", exc)


def _call_responses_api(
    prompt_id: str,
    prompt_version: Optional[str],
    message_text: str,
    metadata: Optional[Dict[str, Any]] = None,
    include_fields: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    if OpenAI is None:
        raise HTTPException(status_code=500, detail="openai package not installed")
    if not prompt_id:
        raise HTTPException(status_code=500, detail="Prompt ID is not configured")

    client = OpenAI(api_key=OPENAI_API_KEY)
    input_block = {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": message_text,
            }
        ],
    }
    prompt_config: Dict[str, Any] = {"id": prompt_id}
    if prompt_version:
        prompt_config["version"] = prompt_version

    body: Dict[str, Any] = {
        "prompt": prompt_config,
        "input": [input_block],
        "reasoning": {},
        "store": STORE_RESPONSES,
    }
    if RESPONSES_MODEL:
        body["model"] = RESPONSES_MODEL
    if VECTOR_STORE_IDS:
        body["tools"] = [
            {
                "type": "file_search",
                "vector_store_ids": VECTOR_STORE_IDS,
            }
        ]
    include = include_fields if include_fields is not None else RESPONSES_INCLUDE_FIELDS
    if include:
        body["include"] = include
    if metadata:
        body["metadata"] = metadata

    response_payload = client.post(
        "/responses",
        cast_to=Dict[str, Any],
        body=body,
    )
    reply_text = _extract_output_text(response_payload)
    if not reply_text:
        reply_text = ""
    return reply_text, response_payload


ROUTE_PATTERN = re.compile(r"ROUTE\s*:\s*(controller|trainer|practicer|coach)", re.IGNORECASE)


def _extract_route_tag(text: str) -> Tuple[str, Optional[str]]:
    if not text:
        return "", None
    match = ROUTE_PATTERN.search(text)
    route_to: Optional[str] = None
    if match:
        route_to = match.group(1).lower()
    cleaned_lines = []
    for line in text.splitlines():
        if ROUTE_PATTERN.search(line):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines).strip() or text.strip()
    return cleaned, route_to


def _google_sheets_available() -> bool:
    return bool(
        gspread
        and Credentials
        and GOOGLE_SERVICE_ACCOUNT_INFO
        and GOOGLE_SHEET_ID
    )


def _require_dependencies(for_login: bool = False) -> None:
    if for_login and bcrypt is None:
        raise HTTPException(status_code=500, detail="bcrypt is not installed; run `pip install -r requirements.txt`.")
    if not _google_sheets_available():
        raise HTTPException(
            status_code=500,
            detail=(
                "Google Sheets credentials are not configured. Set GOOGLE_SERVICE_ACCOUNT_INFO and GOOGLE_SHEET_ID to enable registration."
            ),
        )


@lru_cache(maxsize=1)
def _service_account_info() -> Dict[str, str]:
    raw = GOOGLE_SERVICE_ACCOUNT_INFO
    if not raw:
        raise HTTPException(status_code=500, detail="GOOGLE_SERVICE_ACCOUNT_INFO is required")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        if os.path.exists(raw):
            with open(raw, "r", encoding="utf-8") as f:
                return json.load(f)
        raise HTTPException(status_code=500, detail="GOOGLE_SERVICE_ACCOUNT_INFO must be a JSON string or file path")


@lru_cache(maxsize=1)
def _get_spreadsheet():
    _require_dependencies()
    try:
        credentials = Credentials.from_service_account_info(_service_account_info(), scopes=GOOGLE_SCOPES)
        client = gspread.authorize(credentials)
        return client.open_by_key(GOOGLE_SHEET_ID)
    except GoogleAuthError as exc:  # pragma: no cover - depends on credentials
        raise HTTPException(status_code=500, detail=f"Failed to authenticate with Google Sheets: {exc}")
    except Exception as exc:  # pragma: no cover - unexpected
        raise HTTPException(status_code=500, detail=f"Failed to connect to Google Sheets: {exc}")


def _ensure_worksheet(title: str, headers: list[str]):
    spreadsheet = _get_spreadsheet()
    try:
        worksheet = spreadsheet.worksheet(title)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=title, rows=1000, cols=max(len(headers), 4))
    if rowcol_to_a1 is None:
        raise HTTPException(status_code=500, detail="gspread utilities are unavailable")
    with worksheet_lock:
        existing = [cell.strip().lower() for cell in worksheet.row_values(1)]
        expected = [cell.lower() for cell in headers]
        if existing != expected:
            end_cell = rowcol_to_a1(1, len(headers))
            worksheet.update(f"A1:{end_cell}", [headers])
    return worksheet


def _hash_password(raw_password: str) -> str:
    if bcrypt is None:
        raise HTTPException(status_code=500, detail="bcrypt is not installed")
    return bcrypt.hashpw(raw_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(raw_password: str, password_hash: str) -> bool:
    if bcrypt is None:
        raise HTTPException(status_code=500, detail="bcrypt is not installed")
    try:
        return bcrypt.checkpw(raw_password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:  # pragma: no cover - corrupt hash
        return False


def _get_user_record(email: str) -> Optional[Dict[str, str]]:
    if not _google_sheets_available():
        return None
    try:
        worksheet = _ensure_worksheet(GOOGLE_USERS_WORKSHEET, USERS_HEADERS)
        for row in worksheet.get_all_records(expected_headers=USERS_HEADERS):
            if row.get("email", "").strip().lower() == email.lower():
                return row
    except Exception as exc:  # pragma: no cover - network/API
        logger.warning("Failed to load user record from Google Sheets: %s", exc)
    return None


def _append_user(email: str, full_name: str, password_hash: str) -> None:
    worksheet = _ensure_worksheet(GOOGLE_USERS_WORKSHEET, USERS_HEADERS)
    record = [datetime.utcnow().isoformat(), email, full_name, password_hash]
    with worksheet_lock:
        worksheet.append_row(record, value_input_option="USER_ENTERED")


def _record_interaction(user_id: str, message: str, reply: str) -> None:
    if not _google_sheets_available():
        return
    try:
        worksheet = _ensure_worksheet(GOOGLE_INTERACTIONS_WORKSHEET, INTERACTION_HEADERS)
        entry = [
            datetime.utcnow().isoformat(),
            user_id or "anonymous",
            message,
            reply,
        ]
        with worksheet_lock:
            worksheet.append_row(entry, value_input_option="USER_ENTERED")
    except Exception as exc:  # pragma: no cover - logging failure should not break chat
        logger.warning("Failed to record interaction: %s", exc)


def _encode_user_cookie(email: str, full_name: str) -> str:
    payload = json.dumps({"email": email, "full_name": full_name})
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")


def _extract_output_text(response_payload: Dict[str, object]) -> str:
    """Best-effort extraction of text from a Responses API payload."""
    if not isinstance(response_payload, dict):
        return ""

    # Direct output_text provided as list of chunks or single string
    output_text = response_payload.get("output_text")
    if isinstance(output_text, list):
        combined = "".join(str(item) for item in output_text if isinstance(item, str))
        if combined:
            return combined
    elif isinstance(output_text, str):
        return output_text

    texts: List[str] = []

    def collect_from_content(content: object) -> None:
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            text_value = block.get("text") or block.get("value") or block.get("content")
            if block_type in {"output_text", "text", "message"} and isinstance(text_value, str):
                texts.append(text_value)
            elif block_type == "tool_call" and isinstance(block.get("output"), str):
                texts.append(str(block.get("output")))

    output = response_payload.get("output")
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict):
                collect_from_content(item.get("content"))

    if not texts and isinstance(response_payload.get("message"), dict):
        collect_from_content(response_payload["message"].get("content"))

    if texts:
        return "".join(texts)

    result = response_payload.get("result")
    if isinstance(result, str):
        return result

    return ""


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def serve_root() -> HTMLResponse:
    """Serve the single-page app shell."""
    web_dir = os.path.join(os.path.dirname(__file__), "web")
    index_path = os.path.join(web_dir, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:  # pragma: no cover - cosmetic route
    return Response(status_code=204)


@app.post("/api/register", response_model=AuthResponse)
async def register_user(payload: RegisterRequest) -> AuthResponse:
    _require_dependencies(for_login=True)

    email = (payload.email or "").strip().lower()
    password = (payload.password or "").strip()
    full_name = (payload.full_name or "").strip()

    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="A valid email address is required")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long")

    existing = _get_user_record(email)
    if existing:
        raise HTTPException(status_code=400, detail="An account with this email already exists")

    password_hash = _hash_password(password)
    _append_user(email, full_name, password_hash)

    return AuthResponse(email=email, full_name=full_name)


@app.post("/api/login", response_model=AuthResponse)
async def login_user(payload: LoginRequest, response: Response) -> AuthResponse:
    _require_dependencies(for_login=True)

    email = (payload.email or "").strip().lower()
    password = (payload.password or "").strip()

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")

    user = _get_user_record(email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    password_hash = user.get("password_hash", "")
    if not password_hash or not _verify_password(password, password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    full_name = user.get("full_name", "")
    if payload.remember_me:
        response.set_cookie(
            AUTH_COOKIE_NAME,
            value=_encode_user_cookie(email, full_name),
            max_age=AUTH_COOKIE_MAX_AGE_SECONDS,
            path="/",
            secure=False,
            httponly=False,
            samesite="Lax",
        )
    else:
        response.delete_cookie(AUTH_COOKIE_NAME, path="/")

    return AuthResponse(email=email, full_name=full_name)


@app.post("/api/logout")
async def logout_user(response: Response) -> Dict[str, str]:
    response.delete_cookie(AUTH_COOKIE_NAME, path="/")
    return {"status": "logged_out"}


def _build_controller_context(state: Dict[str, Any], progress: Dict[str, Any]) -> str:
    training = state.get("training_progress", {}) or {}
    completed = training.get("completed_topics", []) or []
    in_progress = training.get("in_progress_topic") or ""
    notes = training.get("notes") or ""
    practice_entries = state.get("practice_history", []) or []
    coaching_entries = state.get("coaching_feedback", []) or []

    # Build progress summary
    completed_topics = progress.get("completed_topics", [])
    completed_practices = progress.get("completed_practices", 0)
    completed_coaching = progress.get("completed_coaching_sessions", 0)
    last_task = progress.get("last_task") or "None"
    last_activity = progress.get("last_activity") or "None"

    context_lines = [
        f"Current role: {state.get('current_role', 'controller')}",
        "",
        "=== USER PROGRESS SUMMARY ===",
        f"Completed training topics: {', '.join(completed_topics) if completed_topics else 'None'}",
        f"Total practice sessions completed: {completed_practices}",
        f"Total coaching sessions completed: {completed_coaching}",
        f"Last activity: {last_activity}",
        f"Last task: {last_task}",
        "",
        "=== CURRENT SESSION ===",
        f"Topic in progress: {in_progress or 'None'}",
    ]
    if notes:
        context_lines.append(f"Training notes: {notes}")
    if practice_entries:
        last_practice = practice_entries[-1]
        context_lines.append(
            "Last practice in this session: "
            + f"Scenario={last_practice.get('scenario', '')}, Difficulty={last_practice.get('difficulty', '')}, Outcome={last_practice.get('outcome', '')}"
        )
    if coaching_entries:
        context_lines.append(f"Recent coaching feedback: {coaching_entries[-1].get('feedback', '')}")

    context_lines.append("")
    context_lines.append("Based on the user's progress, suggest appropriate next steps (training, practice, or coaching).")
    return "\n".join(context_lines)


def _append_history(state: Dict[str, Any], key: str, entry: Dict[str, Any], limit: int = 10) -> None:
    history = state.setdefault(key, [])
    if not isinstance(history, list):
        history = []
    history.append(entry)
    state[key] = history[-limit:]


@app.get("/api/agent/state")
async def agent_state(user_id: str) -> Dict[str, Any]:
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    state, _ = _load_user_state(user_id)
    sanitized = _sanitize_state_for_storage(state)
    return {"state": sanitized}


@app.post("/api/agent/controller", response_model=AgentResponse)
async def agent_controller(payload: ControllerRequest) -> AgentResponse:
    if not payload.user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    state, row = _load_user_state(payload.user_id)
    progress = _load_user_progress(payload.user_id)
    context = _build_controller_context(state, progress)
    instruction = (
        "You are the session controller for an evangelism training assistant."
        " Greet the user warmly and concisely, confirm their request, and guide them toward"
        " the right assistant (trainer, practicer, or coach). After your conversational reply,"
        " add a separate line exactly in the format `ROUTE: <role>` indicating which agent"
        " should handle the next interaction. Valid roles are controller, trainer, practicer, coach."
    )
    message_text = (
        f"{instruction}\n\nContext:\n{context}\n\nUser message:\n{payload.message}"
    )
    reply, response_payload = _call_responses_api(
        CONTROLLER_PROMPT_ID,
        CONTROLLER_PROMPT_VERSION,
        message_text,
        metadata={"user_id": payload.user_id, "agent": "controller"},
    )
    reply, route_from_text = _extract_route_tag(reply)

    timestamp = datetime.utcnow().isoformat()
    _append_history(
        state,
        "controller_history",
        {"timestamp": timestamp, "user": payload.message, "assistant": reply},
    )

    response_meta: Dict[str, Any] = {}
    route_to: Optional[str] = None
    if isinstance(response_payload.get("metadata"), dict):
        response_meta["metadata"] = response_payload["metadata"]
        route_to = response_payload["metadata"].get("route_to")
        if isinstance(route_to, str) and route_to.strip():
            route_to = route_to.strip()
            response_meta["route_to"] = route_to

    if not route_to and route_from_text:
        route_to = route_from_text
        response_meta["route_to"] = route_to

    state["current_role"] = route_to or "controller"

    # Update progress
    progress["last_activity"] = f"controller: {payload.message[:50]}"
    progress["last_task"] = route_to or "controller"
    _save_user_progress(payload.user_id, progress)

    sanitized = _sanitize_state_for_storage(state)
    _save_user_state(payload.user_id, state, row)

    return AgentResponse(reply=reply, state=sanitized, metadata=response_meta)


@app.post("/api/agent/trainer", response_model=AgentResponse)
async def agent_trainer(payload: TrainerRequest) -> AgentResponse:
    if not payload.user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    state, row = _load_user_state(payload.user_id)
    training = state.setdefault("training_progress", _default_user_state()["training_progress"])

    topic_restart = False
    if payload.topic and training.get("in_progress_topic") == payload.topic:
        topic_info = get_topic_info(payload.topic)
        topic_title = topic_info.get("title", "") if isinstance(topic_info, dict) else ""
        normalized_message = payload.message.strip().lower() if payload.message else ""
        candidate_labels = {str(payload.topic).strip().lower()}
        if topic_title:
            candidate_labels.add(topic_title.strip().lower())
            candidate_labels.add(f"{payload.topic}. {topic_title}".strip().lower())
        if normalized_message and (
            normalized_message in candidate_labels
            or normalized_message.startswith(f"{payload.topic}.")
        ):
            topic_restart = True

    if payload.topic and (payload.topic != training.get("in_progress_topic") or topic_restart):
        training["in_progress_topic"] = payload.topic
        training["current_step"] = 1
        training["awaiting_response"] = False

    current_topic = payload.topic or training.get("in_progress_topic")
    current_step = int(training.get("current_step", 1)) if isinstance(training, dict) else 1
    if current_step < 1:
        current_step = 1
        training["current_step"] = current_step

    total_steps = get_total_steps(current_topic) if current_topic else 0
    if current_topic and total_steps and current_step > total_steps:
        current_step = total_steps
        training["current_step"] = current_step

    awaiting_response = bool(training.get("awaiting_response", False)) if isinstance(training, dict) else False
    progress_indicator = ""
    if current_topic:
        progress_indicator = _render_progress_bar(current_step, total_steps, awaiting_response)

    if not current_topic:
        base_message = "Ask the user to select a training topic (1-4)."
        message_text = f"{base_message}\n\nUser message:\n{payload.message}"
    else:
        learner_message = payload.message if awaiting_response else None
        context = build_training_context(current_topic, current_step, learner_message)
        preface_sections: List[str] = []
        if progress_indicator:
            preface_sections.append(
                "PROGRESS INDICATOR:\n"
                f"{progress_indicator}\n"
                "Include this indicator verbatim at the top of your reply. After the indicator, follow the guidance below."
            )
        if awaiting_response:
            preface_sections.append(
                "The learner has just responded to this step. Review their message below. "
                "If it meets the completion signal, acknowledge them briefly and stop. "
                "After the progress indicator, start your acknowledgement with `✅` when the step is complete. "
                "If it does not meet the completion signal, start with `🔄`, give concise coaching, and invite them to try again. "
                "Do NOT introduce the next step yourself."
            )
        else:
            preface_sections.append(
                "Introduce this step using the instructions below. Keep the reply brief and end with the question or prompt for the learner."
            )
        preface_sections.append(context)
        message_text = "\n\n".join(preface_sections)

    metadata_payload = {
        "user_id": payload.user_id,
        "agent": "trainer",
        "topic": current_topic or "",
    }
    if current_topic:
        metadata_payload["step"] = str(current_step)
        metadata_payload["awaiting_response"] = "true" if awaiting_response else "false"
        if progress_indicator:
            metadata_payload["progress_bar"] = progress_indicator
    if current_topic and total_steps:
        metadata_payload["total_steps"] = str(total_steps)

    reply_primary, _ = _call_responses_api(
        TRAINER_PROMPT_ID,
        TRAINER_PROMPT_VERSION,
        message_text,
        metadata=metadata_payload,
    )

    reply_parts: List[str] = [reply_primary]

    needs_new_step_intro = False

    progress = _load_user_progress(payload.user_id)

    step_marked_complete = awaiting_response and _is_step_completed_reply(reply_primary)

    if payload.mark_complete and current_topic:
        completed = training.setdefault("completed_topics", [])
        if current_topic not in completed:
            completed.append(current_topic)
        training["in_progress_topic"] = None
        training["current_step"] = 1
        training["awaiting_response"] = False

        if current_topic not in progress["completed_topics"]:
            progress["completed_topics"].append(current_topic)
        progress["last_activity"] = f"Completed training: {current_topic}"
        progress["last_task"] = "trainer"
        _save_user_progress(payload.user_id, progress)
        _append_training_completion(payload.user_id, current_topic, total_steps)
    else:
        if current_topic:
            if awaiting_response:
                if step_marked_complete:
                    training["awaiting_response"] = False
                    if not is_topic_complete(current_topic, current_step):
                        next_step = current_step + 1
                        if total_steps and next_step > total_steps:
                            next_step = total_steps
                        training["current_step"] = next_step
                        training["in_progress_topic"] = current_topic
                        needs_new_step_intro = True
                    else:
                        training["current_step"] = current_step
                else:
                    training["awaiting_response"] = True
                    training["current_step"] = current_step
                    training["in_progress_topic"] = current_topic
            else:
                training["awaiting_response"] = True
                training["in_progress_topic"] = current_topic

    if current_topic and needs_new_step_intro:
        next_step = int(training.get("current_step", current_step))
        total_steps_after_intro = get_total_steps(current_topic)
        intro_progress_indicator = _render_progress_bar(next_step, total_steps_after_intro, False)
        next_context = build_training_context(current_topic, next_step)
        intro_sections: List[str] = []
        if intro_progress_indicator:
            intro_sections.append(
                "PROGRESS INDICATOR:\n"
                f"{intro_progress_indicator}\n"
                "Include this indicator verbatim at the top of your reply before any other text."
            )
        intro_sections.append(
            "Introduce this step using the instructions below. Keep the reply brief and end with the question or prompt for the learner."
        )
        intro_sections.append(next_context)
        intro_message_text = "\n\n".join(intro_sections)

        intro_metadata = {
            "user_id": payload.user_id,
            "agent": "trainer",
            "topic": current_topic or "",
            "step": str(next_step),
            "total_steps": str(total_steps_after_intro) if total_steps_after_intro else None,
            "awaiting_response": "false",
        }
        if intro_progress_indicator:
            intro_metadata["progress_bar"] = intro_progress_indicator
        # Remove None values
        intro_metadata = {k: v for k, v in intro_metadata.items() if v is not None}

        reply_intro, _ = _call_responses_api(
            TRAINER_PROMPT_ID,
            TRAINER_PROMPT_VERSION,
            intro_message_text,
            metadata=intro_metadata,
        )
        reply_parts.append(reply_intro)
        training["awaiting_response"] = True

    timestamp = datetime.utcnow().isoformat()

    if payload.notes:
        training["notes"] = payload.notes

    state["current_role"] = "trainer"

    updated_step = int(training.get("current_step", current_step))
    if updated_step < 1:
        updated_step = 1
        training["current_step"] = updated_step
    total_steps_after = get_total_steps(current_topic) if current_topic else 0
    awaiting_after = bool(training.get("awaiting_response", False))

    if current_topic and not payload.mark_complete:
        step_desc = f"{updated_step}/{total_steps_after}" if total_steps_after else str(updated_step)
        phase_desc = "awaiting learner response" if awaiting_after else "ready to introduce next step"
        progress["last_activity"] = f"Training: Topic {current_topic}, Step {step_desc} ({phase_desc})"
        progress["last_task"] = "trainer"
        _save_user_progress(payload.user_id, progress)

    response_progress_indicator = ""
    if current_topic and not payload.mark_complete:
        response_progress_indicator = _render_progress_bar(updated_step, total_steps_after, awaiting_after)

    active_topic = training.get("in_progress_topic") or current_topic
    response_metadata: Dict[str, Any] = {
        "topic": active_topic,
        "step": updated_step if (active_topic and not payload.mark_complete) else None,
        "total_steps": total_steps_after if (active_topic and not payload.mark_complete) else None,
        "awaiting_response": awaiting_after if (active_topic and not payload.mark_complete) else False,
        "mark_complete": payload.mark_complete,
    }
    if response_progress_indicator:
        response_metadata["progress_bar"] = response_progress_indicator

    combined_reply = "\n\n".join(part for part in reply_parts if part).strip()

    _append_history(
        state,
        "trainer_history",
        {
            "timestamp": timestamp,
            "user": payload.message,
            "assistant": combined_reply,
            "topic": current_topic,
            "step": updated_step if current_topic else None,
        },
    )

    sanitized = _sanitize_state_for_storage(state)
    _save_user_state(payload.user_id, state, row)

    return AgentResponse(
        reply=combined_reply,
        state=sanitized,
        metadata=response_metadata,
    )


def _format_transcript(transcript: List[AgentMessage], latest_user: str) -> str:
    lines = []
    for turn in transcript:
        lines.append(f"{turn.role.capitalize()}: {turn.content}")
    lines.append(f"User: {latest_user}")
    return "\n".join(lines)


@app.post("/api/agent/practicer", response_model=AgentResponse)
async def agent_practicer(payload: PracticerRequest) -> AgentResponse:
    if not payload.user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    if not payload.scenario:
        raise HTTPException(status_code=400, detail="scenario is required")

    state, row = _load_user_state(payload.user_id)
    transcript_text = _format_transcript(payload.transcript, payload.message)
    context = (
        f"Scenario: {payload.scenario}\n"
        f"Difficulty: {payload.difficulty or 'unspecified'}\n"
        f"Outcome target: {payload.outcome or 'unspecified'}\n"
        f"Conversation so far:\n{transcript_text}"
    )

    message_text = (
        "You are the practice roleplay agent. Stay in character and respond realistically."\
        + f"\n\n{context}"
    )

    reply, _ = _call_responses_api(
        PRACTICER_PROMPT_ID,
        PRACTICER_PROMPT_VERSION,
        message_text,
        metadata={
            "user_id": payload.user_id,
            "agent": "practicer",
            "scenario": payload.scenario,
            "difficulty": payload.difficulty or "",
        },
    )

    timestamp = datetime.utcnow().isoformat()
    updated_transcript = [turn.dict() for turn in payload.transcript]
    updated_transcript.append({"role": "user", "content": payload.message})
    updated_transcript.append({"role": "bot", "content": reply})

    summary_entry = {
        "timestamp": timestamp,
        "scenario": payload.scenario,
        "difficulty": payload.difficulty,
        "outcome": payload.outcome,
        "last_exchange": {"user": payload.message, "assistant": reply},
    }
    _append_history(state, "practicer_history", summary_entry)
    state["current_role"] = "practicer"

    # Update progress tracking
    progress = _load_user_progress(payload.user_id)
    progress["last_activity"] = f"Practice: {payload.scenario}"
    progress["last_task"] = "practicer"

    if payload.session_completed:
        _append_practice_history(
            payload.user_id,
            payload.scenario,
            payload.difficulty,
            updated_transcript,
            payload.outcome,
        )
        # reflect summary in general practice history list for quick reference
        _append_history(state, "practice_history", summary_entry)

        # Increment completed practices count
        progress["completed_practices"] = progress.get("completed_practices", 0) + 1
        progress["last_activity"] = f"Completed practice: {payload.scenario}"

    _save_user_progress(payload.user_id, progress)

    sanitized = _sanitize_state_for_storage(state)
    _save_user_state(payload.user_id, state, row)

    return AgentResponse(
        reply=reply,
        state=sanitized,
        metadata={
            "transcript": updated_transcript,
            "session_completed": payload.session_completed,
        },
    )


@app.post("/api/agent/coach", response_model=AgentResponse)
async def agent_coach(payload: CoachRequest) -> AgentResponse:
    if not payload.user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    if not payload.transcript:
        raise HTTPException(status_code=400, detail="transcript is required")

    state, row = _load_user_state(payload.user_id)
    transcript_text = "\n".join(
        f"{turn.role.capitalize()}: {turn.content}" for turn in payload.transcript
    )
    previous_feedback = payload.previous_feedback or state.get("coaching_feedback", [])

    context = (
        "Provide coaching feedback using the rubric.\n\n"
        f"Transcript:\n{transcript_text}\n\n"
        f"Previous feedback: {previous_feedback or 'None'}\n"
        f"Rubric: {json.dumps(payload.rubric or {}, ensure_ascii=False)}"
    )

    message_text = f"{context}\n\nUser question:\n{payload.message}"

    reply, _ = _call_responses_api(
        COACH_PROMPT_ID,
        COACH_PROMPT_VERSION,
        message_text,
        metadata={"user_id": payload.user_id, "agent": "coach"},
    )

    timestamp = datetime.utcnow().isoformat()
    feedback_entry = {
        "timestamp": timestamp,
        "feedback": reply,
        "next_steps": payload.next_steps,
    }
    _append_history(state, "coach_history", feedback_entry)
    state["current_role"] = "coach"

    _append_coaching_feedback(payload.user_id, reply, payload.next_steps, payload.rubric)
    _append_history(state, "coaching_feedback", feedback_entry)

    # Update progress tracking
    progress = _load_user_progress(payload.user_id)
    progress["completed_coaching_sessions"] = progress.get("completed_coaching_sessions", 0) + 1
    progress["last_activity"] = "Coaching session completed"
    progress["last_task"] = "coach"
    _save_user_progress(payload.user_id, progress)

    sanitized = _sanitize_state_for_storage(state)
    _save_user_state(payload.user_id, state, row)

    return AgentResponse(
        reply=reply,
        state=sanitized,
        metadata={"feedback_timestamp": timestamp},
    )


@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    """Stream chat responses using the Responses API"""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    if not PROMPT_ID:
        raise HTTPException(status_code=500, detail="OPENAI_PROMPT_ID is not configured")

    tools: List[Dict[str, object]] = []
    if VECTOR_STORE_IDS:
        tools.append({
            "type": "file_search",
            "vector_store_ids": VECTOR_STORE_IDS,
        })

    payload: Dict[str, object] = {
        "prompt": {
            "id": PROMPT_ID,
            "version": PROMPT_VERSION,
        },
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": req.message.strip(),
                    }
                ],
            }
        ],
        "reasoning": {},
    }

    if RESPONSES_MODEL:
        payload["model"] = RESPONSES_MODEL
    if tools:
        payload["tools"] = tools
    if RESPONSES_INCLUDE_FIELDS:
        payload["include"] = RESPONSES_INCLUDE_FIELDS
    payload["store"] = STORE_RESPONSES

    async def generate():
        full_reply = ""
        try:
            if OpenAI is None:
                raise RuntimeError("openai package not installed")
            client = OpenAI(api_key=OPENAI_API_KEY)
            response_payload = client.post(
                "/responses",
                cast_to=Dict[str, Any],
                body=payload,
            )

            full_reply = _extract_output_text(response_payload)
            if not full_reply:
                raise RuntimeError("Responses API did not return any text output")

            yield f"data: {json.dumps({'content': full_reply})}\n\n"

            if req.user_id and req.message:
                _record_interaction(req.user_id.strip(), req.message.strip(), full_reply)
            elif req.message:
                _record_interaction("anonymous", req.message.strip(), full_reply)

            yield f"data: {json.dumps({'done': True, 'audio_mode': req.audio_mode})}\n\n"

        except Exception as e:
            logger.exception("chat_stream failed")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)) -> Dict[str, str]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    if OpenAI is None:
        raise HTTPException(status_code=500, detail="openai package not installed")
    client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        # Read bytes and send to transcription
        data = await audio.read()
        from io import BytesIO
        buf = BytesIO(data)
        buf.name = audio.filename or "audio.webm"
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=buf,
        )
        text = transcript.text or ""
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"transcription failed: {e}")


@app.post("/api/realtime/session")
async def create_realtime_session() -> Dict[str, Any]:
    """Create a realtime session for audio chat"""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    # Return the API key for direct WebSocket connection
    # The prompt instructions will be sent via WebSocket after connection
    return {
        "api_key": OPENAI_API_KEY,
        "model": "gpt-4o-realtime-preview-2024-12-17"
    }


@app.post("/api/tts")
async def tts(request: Request) -> Response:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    if OpenAI is None:
        raise HTTPException(status_code=500, detail="openai package not installed")
    client = OpenAI(api_key=OPENAI_API_KEY)

    content_type = request.headers.get("content-type", "")
    voice = "alloy"
    text: Optional[str] = None

    try:
        if "multipart/form-data" in content_type:
            if not HAS_MULTIPART:
                raise HTTPException(
                    status_code=500,
                    detail="python-multipart is required to process form data; either install it or send JSON",
                )
            form = await request.form()
            text = (form.get("text") or "").strip()
            voice = (form.get("voice") or voice).strip() or voice
        else:
            data = await request.json()
            text = str(data.get("text", "")).strip()
            voice = str(data.get("voice", voice)).strip() or voice
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid request body: {e}")

    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    try:
        speech = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="mp3",
        )
        audio_bytes = speech.read()
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"tts failed: {e}")


# Serve static React UI at /app to avoid shadowing /api/* routes
web_dir = os.path.join(os.path.dirname(__file__), "web")
if os.path.isdir(web_dir):
    app.mount("/app", StaticFiles(directory=web_dir, html=True), name="web")
