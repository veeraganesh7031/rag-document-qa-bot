@echo off
call .venv\Scripts\activate
if not exist .env (
  copy .env.example .env
  echo Please add your OPENAI_API_KEY to .env and run this file again.
  pause
  exit /b
)
python ingest.py
streamlit run app.py
pause
