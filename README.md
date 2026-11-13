# agentic-ai-fastapi


py -m venv .venv

.venv\Scripts\activate

pip install "fastapi[standard]"
pip install uvicorn
    uvicorn --version
    fastapi --version

    running server:
    uvicorn main:app --reload
    for swagger
    http://127.0.0.1:8000/docs