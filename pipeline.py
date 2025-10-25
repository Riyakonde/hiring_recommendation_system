from typing import TypedDict
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END

import re
import os

llm = None  # will be initialized in run_pipeline

class HiringState(TypedDict):
    resume_path: str
    job_path: str
    resume_text: str
    job_text: str
    resume_summary: str
    job_summary: str
    skills_score: float
    experience_score: float
    education_score: float
    overall_score: float
    decision: str
    recommendation: str
    final_report: str
    candidate_name: str  # NEW: store candidate name

str_parser = StrOutputParser()


# --- Node functions ---

def load_resume(state: HiringState):
    loader = PyMuPDFLoader(state["resume_path"])
    docs = loader.load()
    state["resume_text"] = "\n".join([doc.page_content for doc in docs])
    return state

def extract_candidate_name(state: HiringState) -> HiringState:
    """Extract candidate name from resume using LLM"""
    name_prompt = PromptTemplate.from_template("""
Extract the candidate’s full name from the following resume text.
Respond ONLY with the name.
Resume:
{resume_text}
""")
    name_chain = name_prompt | llm | str_parser
    state["candidate_name"] = name_chain.invoke({"resume_text": state["resume_text"]}).strip()
    print("Extracted candidate name:", state["candidate_name"])
    return state

def load_job(state: HiringState):
    loader = PyMuPDFLoader(state["job_path"])
    docs = loader.load()
    state["job_text"] = "\n".join([doc.page_content for doc in docs])
    return state

def resume_summary(state: HiringState):
    resume_prompt = PromptTemplate.from_template("""
Summarize the resume clearly in sections:
  Skills:
  Experience:
  Education:
  Achievements:
Resume: {resume_text}
""")
    resume_chain = resume_prompt | llm | str_parser
    state["resume_summary"] = resume_chain.invoke({"resume_text": state["resume_text"]})
    return state

def job_description_summary(state: HiringState):
    job_prompt = PromptTemplate.from_template("""
Summarize the job description in sections:
  Required skills:
  Experience level:
  Education requirements:
  Key responsibilities:
Job description: {job_text}
""")
    job_chain = job_prompt | llm | str_parser
    state["job_summary"] = job_chain.invoke({"job_text": state["job_text"]})
    return state



def compare_job_resume(state: HiringState):
    """
    Compare resume and job description summaries using LLM for individual scores,
    then compute a deterministic overall score.
    """
    score_prompt = PromptTemplate.from_template("""
Compare the following resume summary and job description summary.
Provide a JSON formatted response with scores 0–100 for each category:
{{
  "skills_score": <num>,
  "experience_score": <num>,
  "education_score": <num>
}}
Do NOT provide an overall score — it will be computed automatically.
Resume summary: {resume_summary}
Job summary: {job_summary}
""")

    score_chain = score_prompt | llm | JsonOutputParser()
    scores = score_chain.invoke({
        "resume_summary": state["resume_summary"],
        "job_summary": state["job_summary"]
    })

    # Ensure each component score exists and is numeric
    for key in ["skills_score", "experience_score", "education_score"]:
        scores[key] = float(scores.get(key, 0))

    # Compute deterministic overall score using the defined weights
    scores["overall_score"] = (
        scores["skills_score"] * 0.5 +
        scores["experience_score"] * 0.3 +
        scores["education_score"] * 0.2
    )

    # Update state
    state.update(scores)
    print("Deterministic Scores:", scores)
    return state


def decide_interview(state: HiringState):
    score = state["overall_score"]
    if score >= 85:
        state["decision"] = "execute_one_interview"
    elif score >= 60:
        state["decision"] = "execute_two_interviews"
    else:
        state["decision"] = "execute_rejection"
    print(f"Decision based on score {score}: {state['decision']}")
    return state

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_report(state: HiringState) -> HiringState:
    recommendation = state.get("recommendation", "Not Available")
    candidate_name = state.get("candidate_name", "Unknown Candidate")

    # Full report prompt in plain text format
    report_prompt = PromptTemplate.from_template("""
You are an HR assistant preparing a professional hiring recommendation report.
Write the report neatly formatted in plain text (under 200 words). Use headings and separators,
no Markdown styling. Use a clipboard emoji 📋 at the top and a separator line under it.

Include the following details:
- Candidate name
- Each score (Skills, Experience, Education, Overall)
- Final recommendation
- A short, clear summary paragraph explaining why this recommendation fits.

Format example:
📋 FINAL HIRING RECOMMENDATION REPORT
-------------------------------------
Candidate: John Doe
Skills Score: 88/100
Experience Score: 76/100
Education Score: 92/100
Overall Score: 84.2

Recommendation: Two Interviews – Initial screening + coding round

Summary:
Candidate demonstrates strong alignment with required skills, education, and experience.
Recommended next steps.

Scores:
Skills Score: {skills_score}/100
Experience Score: {experience_score}/100
Education Score: {education_score}/100
Overall Score: {overall_score}/100
Recommendation: {recommendation}
Candidate: {candidate_name}
""")

    report_chain = report_prompt | llm | StrOutputParser()

    report_text = report_chain.invoke({
        "skills_score": state.get("skills_score", 0),
        "experience_score": state.get("experience_score", 0),
        "education_score": state.get("education_score", 0),
        "overall_score": state.get("overall_score", 0),
        "recommendation": recommendation,
        "candidate_name": candidate_name
    })

    state["final_report"] = report_text.strip()
    return state


def execute_one_interview(state: HiringState) -> HiringState:
    state["recommendation"] = "One Interview – Direct Human Interview"
    state = generate_report(state)
    for key in ["resume_summary", "job_summary", "resume_text", "job_text"]:
        state.pop(key, None)
    return state

def execute_two_interviews(state: HiringState) -> HiringState:
    state["recommendation"] = "Two Interviews – Screening + Coding Round"
    state = generate_report(state)
    for key in ["resume_summary", "job_summary", "resume_text", "job_text"]:
        state.pop(key, None)
    return state

def execute_rejection(state: HiringState) -> HiringState:
    state["recommendation"] = "Rejected – Insufficient match"
    state = generate_report(state)
    for key in ["resume_summary", "job_summary", "resume_text", "job_text"]:
        state.pop(key, None)
    return state

# --- Workflow graph setup ---
workflow = StateGraph(HiringState)
workflow.add_node("load_resume", load_resume)
workflow.add_node("extract_candidate_name", extract_candidate_name)
workflow.add_node("load_job", load_job)
workflow.add_node("resume_summary", resume_summary)
workflow.add_node("job_description_summary", job_description_summary)
workflow.add_node("compare_job_resume", compare_job_resume)
workflow.add_node("decide_interview", decide_interview)
workflow.add_node("generate_report", generate_report)
workflow.add_node("execute_one_interview", execute_one_interview)
workflow.add_node("execute_two_interviews", execute_two_interviews)
workflow.add_node("execute_rejection", execute_rejection)

workflow.add_edge(START, "load_resume")
workflow.add_edge("load_resume", "extract_candidate_name")
workflow.add_edge("extract_candidate_name", "load_job")
workflow.add_edge("load_job", "resume_summary")
workflow.add_edge("resume_summary", "job_description_summary")
workflow.add_edge("job_description_summary", "compare_job_resume")
workflow.add_edge("compare_job_resume", "decide_interview")
workflow.add_edge("decide_interview", "generate_report")
workflow.add_conditional_edges(
    "generate_report",
    lambda s: s["decision"],
    {
        "execute_one_interview": "execute_one_interview",
        "execute_two_interviews": "execute_two_interviews",
        "execute_rejection": "execute_rejection"
    }
)
for node in ["execute_one_interview", "execute_two_interviews", "execute_rejection"]:
    workflow.add_edge(node, END)

graph = workflow.compile()

# --- Run pipeline ---
def run_pipeline(openai_key, resume_path, job_path):
    global llm
    os.environ["OPENAI_API_KEY"] = openai_key
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, temperature=0)
    initial_state = {"resume_path": resume_path, "job_path": job_path}
    final_state = graph.invoke(initial_state)
    return final_state.get("final_report", "No report generated.")
