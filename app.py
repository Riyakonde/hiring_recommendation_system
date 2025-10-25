import streamlit as st
from pipeline import run_pipeline
import os

 
st.set_page_config(page_title="AI Hiring Recommendation System", layout="wide")

 
st.title(" AI Hiring Recommendation System")
st.markdown("""
Welcome! 👋  
This application analyzes a **candidate's resume** and a **job description** using AI (LangChain + LangGraph).  
It compares key skills, experience, and education — then generates a **professional hiring recommendation report**.
""")

 
st.sidebar.header("🔐 Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
st.sidebar.markdown(
    """
    *Your key is never stored or shared.*  
     
    """
)

 
st.subheader("📄 Upload Required Files")
resume_file = st.file_uploader("Upload Candidate Resume (PDF)", type=["pdf"])
job_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

 
if st.button("Generate Hiring Report"):
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar.")
    elif not resume_file or not job_file:
        st.warning("⚠️ Please upload both the Resume and Job Description files.")
    else:
        
        resume_path = "uploaded_resume.pdf"
        job_path = "uploaded_job.pdf"

        with open(resume_path, "wb") as f:
            f.write(resume_file.read())
        with open(job_path, "wb") as f:
            f.write(job_file.read())

        st.info("⏳ Analyzing documents... Please wait a moment...")

        try:
            
            report = run_pipeline(api_key, resume_path, job_path)

            st.success("✅ Hiring Recommendation Report Generated!")
            st.text_area("📋 Final Report", report, height=400)

            
            st.download_button(
                label="💾 Download Report",
                data=report,
                file_name="hiring_recommendation_report.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"⚠️ Error occurred: {e}")

        finally:
             
            if os.path.exists(resume_path):
                os.remove(resume_path)
            if os.path.exists(job_path):
                os.remove(job_path)
