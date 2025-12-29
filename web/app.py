"""FastAPI web application for PyTorch Tensor Marathon."""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from torch_tensor_marathon.problems import initialize_problems
from torch_tensor_marathon.problem import problem_bank, Problem
from torch_tensor_marathon.checker import CorrectnessChecker
from torch_tensor_marathon.gemini_client import GeminiClient

app = FastAPI(title="PyTorch Tensor Marathon", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Initialize problems on startup
initialize_problems()
checker = CorrectnessChecker()
gemini_client = GeminiClient()


class CodeSubmission(BaseModel):
    """Model for code submission."""
    problem_id: str
    user_code: str


class CheckResponse(BaseModel):
    """Model for check response."""
    is_correct: bool
    message: str
    error_type: Optional[str] = None
    expected_shape: Optional[tuple] = None
    actual_shape: Optional[tuple] = None
    execution_output: Optional[str] = None


@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse("web/static/index.html")


@app.get("/api/categories")
async def get_categories():
    """Get all problem categories with statistics."""
    categories = problem_bank.get_categories()
    result = []

    for cat in categories:
        stats = problem_bank.get_category_stats(cat)
        result.append({
            "id": cat,
            "stats": stats
        })

    return {
        "categories": result,
        "total_problems": len(problem_bank)
    }


@app.get("/api/problems/{category}")
async def get_problems_by_category(category: str):
    """Get all problems in a category."""
    problems = problem_bank.get_problems_by_category(category)

    if not problems:
        raise HTTPException(status_code=404, detail="Category not found")

    return {
        "category": category,
        "problems": [
            {
                "id": p.id,
                "title_ja": p.title_ja,
                "title_en": p.title_en,
                "difficulty": p.difficulty,
                "tags": p.tags
            }
            for p in problems
        ]
    }


@app.get("/api/problem/{problem_id}")
async def get_problem(problem_id: str):
    """Get detailed information about a specific problem."""
    problem = problem_bank.get_problem(problem_id)

    if not problem:
        raise HTTPException(status_code=404, detail="Problem not found")

    return {
        "id": problem.id,
        "category": problem.category,
        "difficulty": problem.difficulty,
        "title_ja": problem.title_ja,
        "title_en": problem.title_en,
        "description_ja": problem.description_ja,
        "description_en": problem.description_en,
        "hint_ja": problem.hint_ja,
        "hint_en": problem.hint_en,
        "setup_code": problem.setup_code,
        "template_code": problem.template_code,
        "tags": problem.tags
    }


@app.post("/api/check", response_model=CheckResponse)
async def check_solution(submission: CodeSubmission):
    """Check user's solution."""
    problem = problem_bank.get_problem(submission.problem_id)

    if not problem:
        raise HTTPException(status_code=404, detail="Problem not found")

    result = checker.check_solution(
        setup_code=problem.setup_code,
        user_code=submission.user_code,
        solution_code=problem.solution_code,
        expected_shape=problem.expected_shape
    )

    return CheckResponse(
        is_correct=result.is_correct,
        message=result.message,
        error_type=result.error_type,
        expected_shape=result.expected_shape,
        actual_shape=result.actual_shape,
        execution_output=result.execution_output
    )


@app.get("/api/solution/{problem_id}")
async def get_solution(problem_id: str):
    """Get the solution for a problem."""
    problem = problem_bank.get_problem(problem_id)

    if not problem:
        raise HTTPException(status_code=404, detail="Problem not found")

    return {
        "problem_id": problem_id,
        "solution_code": problem.solution_code
    }


@app.get("/api/stats")
async def get_stats():
    """Get overall statistics."""
    categories = problem_bank.get_categories()
    total_by_difficulty = {
        "beginner": 0,
        "intermediate": 0,
        "advanced": 0,
        "expert": 0
    }

    for cat in categories:
        stats = problem_bank.get_category_stats(cat)
        for level in total_by_difficulty:
            total_by_difficulty[level] += stats[level]

    return {
        "total_problems": len(problem_bank),
        "total_categories": len(categories),
        "by_difficulty": total_by_difficulty
    }


# Gemini API Models
class ApiKeyRequest(BaseModel):
    """Request model for setting API key."""
    api_key: str


class VariationRequest(BaseModel):
    """Request model for problem variation."""
    problem_id: str
    language: str = "en"


class ExplanationRequest(BaseModel):
    """Request model for solution explanation."""
    problem_id: str
    language: str = "en"
    user_code: Optional[str] = None


class HintRequest(BaseModel):
    """Request model for AI hint."""
    problem_id: str
    language: str = "en"
    user_code: Optional[str] = None


# Gemini API Endpoints
@app.post("/api/gemini/set-key")
async def set_gemini_key(request: ApiKeyRequest):
    """Set Gemini API key dynamically."""
    global gemini_client
    gemini_client = GeminiClient(api_key=request.api_key)
    return {"success": True, "enabled": gemini_client.enabled}


@app.get("/api/gemini/enabled")
async def check_gemini_enabled():
    """Check if Gemini API is enabled."""
    return {"enabled": gemini_client.enabled}





@app.post("/api/gemini/hint")
async def generate_hint(request: HintRequest):
    """Generate an AI-powered hint."""
    if not gemini_client.enabled:
        raise HTTPException(status_code=503, detail="Gemini API not enabled")

    problem = problem_bank.get_problem(request.problem_id)
    if not problem:
        raise HTTPException(status_code=404, detail="Problem not found")

    title = problem.title_ja if request.language == "ja" else problem.title_en
    description = problem.description_ja if request.language == "ja" else problem.description_en

    try:
        hint = gemini_client.generate_hint(
            problem_title=title,
            problem_description=description,
            setup_code=problem.setup_code,
            user_code=request.user_code,
            language=request.language
        )

        if not hint:
            raise HTTPException(status_code=500, detail="Failed to generate hint: Empty response")

        return {"hint": hint}
    except Exception as e:
        print(f"Gemini API Error (Hint): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
