# BAM Training Assistant

This project powers the Business & Mission (BAM) multi-agent training assistant. It provides a FastAPI backend with Google Sheets persistence and a single-page React UI for controller, trainer, practicer, and coach agent workflows.

## Prerequisites

- Python 3.8 or newer
- Pip / virtual environment tooling (`python -m venv`)
- Google service-account credentials with edit access to the target spreadsheet
- OpenAI API key with access to the Responses API and the relevant stored prompts

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# copy .env.example to .env and fill in secrets
uvicorn server:app --host 0.0.0.0 --port 8000 --env-file .env
```

Visit `http://localhost:8000` to load the UI.

## Environment Variables

At minimum set:

```
OPENAI_API_KEY=...
OPENAI_CONTROLLER_PROMPT_ID=...
OPENAI_TRAINER_PROMPT_ID=...
OPENAI_PRACTICER_PROMPT_ID=...
OPENAI_COACH_PROMPT_ID=...
GOOGLE_SERVICE_ACCOUNT_INFO=...  # JSON string or absolute path
GOOGLE_SHEET_ID=...
```

Optional overrides include `OPENAI_RESPONSES_MODEL`, prompt versions, and custom sheet tab names (`GOOGLE_SHEET_*`).

## Google Sheets Requirements

Share the destination spreadsheet with the service-account email. The backend will automatically create and maintain the following tabs if they do not exist:

- `Users`
- `Interactions`
- `AgentState`
- `PracticeHistory`
- `CoachingFeedback`

## Deployment Notes

For production, run the app behind a process manager (e.g., `systemd` or a PaaS like Render/Railway) and a reverse proxy (e.g., nginx). Ensure secrets are provided as environment variables, not committed to source control.

## License

Proprietary – internal use only unless otherwise specified.
