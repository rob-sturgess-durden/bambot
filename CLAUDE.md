# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A FastAPI-based multi-agent training assistant for Business & Mission (BAM) evangelism training. The application implements a controller-router pattern with four specialized agents (controller, trainer, practicer, coach) that guide users through training workflows. State is persisted to Google Sheets, and the system uses OpenAI's Responses API with stored prompts.

## Development Commands

### Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the Server
```bash
# Development mode
uvicorn server:app --host 0.0.0.0 --port 8000 --env-file .env --reload

# Production mode (via Passenger WSGI)
# passenger_wsgi.py loads .env and imports server.app as application
```

### Testing
No test suite is currently present in the repository.

## Configuration

Copy `.env.example` to `.env` and configure:
- **Required**: `OPENAI_API_KEY`, agent prompt IDs (CONTROLLER/TRAINER/PRACTICER/COACH_PROMPT_ID), `GOOGLE_SERVICE_ACCOUNT_INFO`, `GOOGLE_SHEET_ID`
- **Optional**: `OPENAI_RESPONSES_MODEL`, prompt versions, custom Google Sheets tab names

`GOOGLE_SERVICE_ACCOUNT_INFO` can be either a JSON string or an absolute path to a credentials file.

## Architecture

### Multi-Agent System
The application routes user interactions through four specialized agents:

1. **Controller** (`/api/agent/controller`): Entry point that greets users and routes to appropriate agents. Extracts routing directives from response text (e.g., `ROUTE: trainer`) or metadata.

2. **Trainer** (`/api/agent/trainer`): Delivers training content on topics. Tracks `training_progress` including completed topics, in-progress topic, and notes. Supports marking topics as complete.

3. **Practicer** (`/api/agent/practicer`): Roleplay agent for scenario-based practice. Maintains conversation transcripts and records completed sessions to `PracticeHistory` sheet.

4. **Coach** (`/api/agent/coach`): Provides feedback on practice transcripts using rubrics. Records feedback to `CoachingFeedback` sheet.

Each agent endpoint accepts a `user_id`, message, and agent-specific context (e.g., topic for trainer, scenario/transcript for practicer).

### State Management
User state is stored per-user in the `AgentState` Google Sheets tab and includes:
- `current_role`: Which agent should handle the next interaction
- Per-agent conversation histories (limited to last 10 entries)
- `training_progress`: completed topics, in-progress topic, notes
- `practice_history` and `coaching_feedback` summaries

State loading/saving functions: `_load_user_state()`, `_save_user_state()`, `_sanitize_state_for_storage()`

### OpenAI Integration
Uses OpenAI Responses API (via `client.post("/responses", ...)`) with stored prompts. Each agent has its own configurable prompt ID and version. Responses are extracted using `_extract_output_text()` which handles various response payload structures.

File search is enabled via `VECTOR_STORE_IDS` configuration.

### Google Sheets Persistence
Five worksheets are auto-created if missing:
- `Users`: Registration/login records with bcrypt password hashes
- `Interactions`: All user-assistant message pairs
- `AgentState`: Per-user state JSON blobs
- `PracticeHistory`: Completed practice session transcripts
- `CoachingFeedback`: Coach feedback with rubrics

Thread-safe writes using `worksheet_lock`. Functions: `_ensure_worksheet()`, `_append_practice_history()`, `_append_coaching_feedback()`

### Authentication
Basic cookie-based auth using bcrypt for password hashing. Endpoints: `/api/register`, `/api/login`, `/api/logout`. Cookie name: `bm_user` (30-day TTL).

### Frontend
Single-page React app served from `web/index.html`. Root path `/` serves the HTML, static assets mounted at `/app`.

## Key Implementation Details

- **Route extraction**: Controller responses may include a `ROUTE: <role>` directive in the text or `metadata.route_to` in the response payload. The `_extract_route_tag()` function strips these directives from user-facing text.

- **History trimming**: All conversation histories are trimmed to the last 10 entries in `_sanitize_state_for_storage()` to prevent state bloat.

- **Transcript formatting**: Practice and coach agents format multi-turn conversations using `_format_transcript()` which combines `AgentMessage` objects with the latest user message.

- **Optional dependencies**: `bcrypt` and `gspread`/`google-auth` are optional. The app raises HTTPException 500 when endpoints requiring these dependencies are called without them installed.

## Common Workflows

### Adding a New Agent
1. Define request/response models (inherit from BaseModel)
2. Add environment variables for prompt ID/version
3. Implement agent endpoint following the pattern: load state → build context → call Responses API → append history → save state
4. Update `_default_user_state()` and `_sanitize_state_for_storage()` if new state fields are needed
5. Update controller prompt to recognize the new agent for routing

### Modifying State Schema
1. Update `_default_user_state()` with new fields
2. Update `_sanitize_state_for_storage()` to handle new fields (especially limits/trimming)
3. Update `_load_user_state()` if deep merging is needed for nested structures
4. Consider migrating existing state in Google Sheets if breaking changes are introduced

### Adding Google Sheets Logging
1. Define headers constant (e.g., `NEW_WORKSHEET_HEADERS`)
2. Add optional env var for tab name (e.g., `GOOGLE_SHEET_NEW_TAB`)
3. Create `_append_*` helper function that calls `_ensure_worksheet()` and uses `worksheet_lock`
4. Call from relevant agent endpoint after successful operation
