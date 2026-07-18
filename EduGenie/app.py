import os
import re
from functools import lru_cache
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import pipeline

load_dotenv()

app = FastAPI(title="EduGenie - Gemini Powered Learning Assistant")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class GroqModel:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.client = None

        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1",
            )

    def available(self):
        return self.client is not None

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are EduGenie, a helpful AI educational assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content or "No response generated."
    

class LearningRequest(BaseModel):
    task: str
    query: str
    level: str = "beginner"


class LearningResponse(BaseModel):
    task: str
    title: str
    output: str
    model_used: str


TASK_TITLES = {
    "question_answering": "Question Answering",
    "explanation": "Concept Explanation",
    "quiz": "Quiz Generation",
    "summary": "Text Summarization",
    "recommendation": "Learning Recommendations",
}


def clean_input(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def format_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def build_prompt(task: str, query: str, level: str) -> str:
    base = (
        "You are EduGenie, a friendly educational assistant. "
        "Give accurate, structured, student-friendly answers. "
        "Use simple language for beginners and deeper explanations for advanced learners."
    )

    if task == "question_answering":
        return f"""
{base}

Task: Answer the student's question.
Student level: {level}
Question: {query}

Format:
1. Direct Answer
2. Key Points
3. Example
"""

    if task == "explanation":
        return f"""
{base}

Task: Explain this concept clearly.
Student level: {level}
Concept: {query}

Format:
1. Simple Definition
2. Step-by-Step Explanation
3. Real-Life Example
4. Common Mistake to Avoid
"""

    if task == "quiz":
        return f"""
{base}

Task: Generate a quiz.
Student level: {level}
Topic: {query}

Create 5 multiple-choice questions.
For each question include:
- Question
- Four options
- Correct answer
- Short explanation
"""

    if task == "summary":
        return f"""
{base}

Task: Summarize this educational content.
Student level: {level}
Content: {query}

Format:
1. Short Summary
2. Important Terms
3. Exam-Oriented Points
"""

    if task == "recommendation":
        return f"""
{base}

Task: Give personalized learning recommendations.
Student level: {level}
Learning goal or difficulty: {query}

Format:
1. What to Learn First
2. Practice Plan
3. Recommended Resources
4. Next Steps
"""

    raise ValueError("Invalid task selected.")


class GeminiModel:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        self.model = None

        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)

    def available(self):
        return self.model is not None

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text or "No response generated."


@lru_cache(maxsize=1)
def get_lamini_model():
    model_name = os.getenv("LOCAL_MODEL", "MBZUAI/LaMini-Flan-T5-783M")
    return pipeline("text2text-generation", model=model_name, max_new_tokens=512)


def generate_with_lamini(prompt: str) -> str:
    generator = get_lamini_model()
    result = generator(prompt)
    return result[0]["generated_text"]


groq = GroqModel()


def run_education_pipeline(task: str, query: str, level: str) -> LearningResponse:
    query = clean_input(query)

    if len(query) < 3:
        return LearningResponse(
            task=task,
            title="Input Error",
            output="Please enter a valid question, topic, or content.",
            model_used="Validation",
        )

    prompt = build_prompt(task, query, level)

    try:
        if groq.available():
         output = groq.generate(prompt)
         model_used = "Groq API"
        else:
         output = generate_with_lamini(prompt)
         model_used = "LaMini-Flan-T5"
    except Exception as error:
        output = (
            "Could not generate the response.\n\n"
            f"Reason: {error}\n\n"
            "Check your API key, internet connection, or installed packages."
        )
        model_used = "Error Handler"

    return LearningResponse(
        task=task,
        title=TASK_TITLES.get(task, "EduGenie Response"),
        output=format_output(output),
        model_used=model_used,
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": None,
            "task": "question_answering",
            "query": "",
            "level": "beginner",
        },
    )

@app.post("/", response_class=HTMLResponse)
def web_generate(
    request: Request,
    task: str = Form(...),
    query: str = Form(...),
    level: str = Form("beginner"),
):
    result = run_education_pipeline(task, query, level)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": result,
            "task": task,
            "query": query,
            "level": level,
        },
    )

@app.post("/api/learn", response_model=LearningResponse)
def api_generate(payload: LearningRequest):
    return run_education_pipeline(payload.task, payload.query, payload.level)


@app.get("/health")
def health():
    return {"status": "ok"}