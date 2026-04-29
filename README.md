# 📄 AI Resume Analyzer

## 📌 Project Overview
This project compares a resume with a job description and generates a match score using NLP techniques.

It also identifies matching skills, missing skills, and gives feedback to improve resume-job alignment.

---

## 🧠 Business Problem
Recruiters receive many resumes and need a faster way to screen candidates.

This project helps:
- Compare resumes with job descriptions
- Identify missing skills
- Generate resume-job match score
- Provide improvement feedback

---

## ⚙️ Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- NLP
- TF-IDF
- Cosine Similarity

---

## 🔄 Project Workflow
1. Load resume and job description text
2. Clean text data
3. Convert text into TF-IDF vectors
4. Calculate cosine similarity
5. Extract skills using predefined skill list
6. Identify missing skills
7. Generate feedback
8. Deploy using Streamlit

---

## 🚀 Streamlit App Features
- Resume text input
- Job description text input
- Match score calculation
- Missing skills detection
- Feedback generation
- Download report

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py