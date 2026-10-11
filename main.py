
import os
import sys
import json
import traceback
from io import StringIO

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Code Interpreter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: list[int]


def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()

    try:
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer

        exec(compile(code, "<string>", "exec"), {})

        output = stdout_buffer.getvalue() + stderr_buffer.getvalue()
        return {"success": True, "output": output}

    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def analyze_error_with_ai(code: str, error_traceback: str) -> list[int]:
    token = os.environ.get("AIPIPE_TOKEN")
    if not token:
        raise RuntimeError("AIPIPE_TOKEN is not configured")

    client = OpenAI(
        api_key=token,
        base_url="https://aipipe.org/openai/v1",
    )

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": (
                    "Analyze Python execution tracebacks. "
                    "Return JSON with one key, error_lines, containing "
                    "the source-code line numbers where the error occurred. "
                    "Count lines in the submitted code starting from 1. "
                    "Use the traceback and do not invent line numbers."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"PYTHON CODE:\n{code}\n\n"
                    f"TRACEBACK:\n{error_traceback}\n\n"
                    'Return JSON only: {"error_lines": [3]}'
                ),
            },
        ],
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)
    analysis = ErrorAnalysis.model_validate(data)
    return sorted(set(
        line for line in analysis.error_lines
        if 1 <= line <= len(code.splitlines())
    ))


@app.get("/")
def home():
    return {"message": "Code Interpreter API is running"}


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return {"error": [], "result": execution["output"]}

    try:
        error_lines = analyze_error_with_ai(
            request.code, execution["output"]
        )
    except Exception:
        # Fallback: extract the source line from the traceback.
        import re
        matches = re.findall(
            r'File "<string>", line (\d+)',
            execution["output"],
        )
        error_lines = [int(matches[-1])] if matches else []

    return {
        "error": error_lines,
        "result": execution["output"],
    }
