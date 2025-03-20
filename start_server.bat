@echo off
echo Starting Python server...

REM Activate virtual environment if not already activated
if not defined VIRTUAL_ENV (
    call venv\Scripts\activate
)

REM Start the Flask server
python app.py 