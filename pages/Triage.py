import os
import json
import streamlit as st
from dotenv import load_dotenv
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem
from reportlab.platypus import Preformatted
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.lib.pagesizes import A4


load_dotenv(".env")

st.set_page_config(
    page_title="Clinical Triage Report",
    page_icon="🩺",
    layout="wide"
)

#---------------Session State Initialization---------------
if "show_summary" not in st.session_state:
    st.session_state.show_summary = False

if "show_detailed" not in st.session_state:
    st.session_state.show_detailed = False
# ---------------- STYLING ----------------
st.markdown("""
<style>

/* FORCE FULL WIDTH */
.main .block-container{
    max-width: 100% !important;
    padding-top: 0.5rem !important;
    padding-left: 0.5rem !important;
    padding-right: 0.5rem !important;
    padding-bottom: 2rem !important;
}

/* Background same as chatbot */
body, .stApp {
    background-color: #a5e6d5;
}

/* Wrapper */
.landing-wrapper{
    width: 100%;
    max-width: 100%;
    margin: 0;
    padding: 0 0.5rem;
}

/* FULL WIDTH DARK HEADER */
.top-box{
    background: #032B63;
    color: white;
    padding: 50px 60px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 25px;
    width: 100%;
}

/* GREEN BOX */
.green-box{
    background: #6FB449;
    color: white;
    padding: 40px 60px;
    border-radius: 15px;
    margin-bottom: 25px;
    width: 100%;
}

/* LIGHT BLUE BOX */
.blue-box{
    background: #0073B4;
    color: white;
    padding: 40px 60px;
    border-radius: 15px;
    margin-bottom: 40px;
    width: 100%;
}

.big-title{
    font-size: 38px;
    font-weight: 900;
    margin-bottom: 15px;
}

.sub-title{
    font-size: 22px;
    font-weight: 600;
}

.text{
    font-size: 17px;
    line-height: 1.7;
}

/* TRIAGE REPORT SECTION */
.triage-header {
    background: #032B63;
    color: white;
    padding: 28px 32px;
    border-radius: 18px;
    font-size: 28px;
    font-weight: 800;
    margin-top: 30px;
    margin-bottom: 20px;
}

.triage-box {
    background: #e9f8f4;
    border-left: 6px solid #0073B4;
    padding: 24px 28px;
    border-radius: 16px;
    line-height: 1.7;
    font-size: 17px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- TRIAGE LANDING HEADER ----------------
st.markdown("""
<div class="landing-wrapper">

<div class="top-box">
<h1 class="big-title">🩺 Intelligent Clinical Triage Assistant</h1>
<p class="sub-title">
From Symptom Understanding to Structured Clinical Guidance — Safe, Calm, and Action-Oriented.
</p>
</div>

<div class="green-box">
<p class="text">
This triage module analyzes your conversation with the AI assistant
and organizes your symptoms into a structured clinical summary.
It evaluates risk level, prioritizes safety, and guides you with
calm, responsible next-step recommendations — without diagnosing.
</p>
</div>



<div class="blue-box">
<p class="text">
This page presents your personalized triage report in a focused
clinical format. It highlights key symptoms, assesses severity,
suggests appropriate home-care strategies, and recommends further
medical evaluation only when necessary.
</p>
</div>

</div>
""", unsafe_allow_html=True)



# ---------------- LOAD SESSION ID ----------------
session_id = st.session_state.get("session_id")

if not session_id:
    st.error("❌ Missing triage session ID.")
    st.stop()

# ---------------- LOAD TRIAGE DATA ----------------
TRIAGE_DIR = "triage_sessions"
file_path = os.path.join(TRIAGE_DIR, f"{session_id}.json")

if not os.path.exists(file_path):
    st.error("❌ Triage session not found.")
    st.stop()

with open(file_path, "r") as f:
    triage_data = json.load(f)

messages = triage_data["messages"]
last_assistant_reply = triage_data["last_assistant_reply"]
model_choice = triage_data["model_choice"]

#------------------------------------------------------------------
import pandas as pd

patient_id = triage_data.get("patient_id")

patients_df = pd.read_csv("patients.csv")

patient_row = patients_df[patients_df["patient_id"] == patient_id]

if not patient_row.empty:
    patient_name = patient_row.iloc[0]["patient_name"]
    patient_age = patient_row.iloc[0]["age"]
    patient_sex = patient_row.iloc[0]["sex"]
else:
    patient_name = "Unknown"
    patient_age = "-"
    patient_sex = "-"


# ---------------- MODEL HELPERS ----------------
import google.generativeai as genai
from groq import Groq

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_reply(model_choice, messages):
    if model_choice.startswith("Gemini"):
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = "\n\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
        return model.generate_content(prompt).text
    else:
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.25,
        )
        return resp.choices[0].message.content
    
#------------------Summary Button---------------------
if not st.session_state.show_summary:
    if st.button("🩺 Generate Triage Summary"):
        st.session_state.show_summary = True
        st.rerun()

# ---------------- GENERATE SUMMARY ----------------
if st.session_state.show_summary:

    summary_prompt = f"""
    Provide a short, calm triage summary (5-6 bullet points max).
    Do not diagnose. Focus on reassurance and simple next steps.

    Patient:
    {last_assistant_reply}
    """

    summary_result = generate_reply(
        model_choice,
        [{"role": "user", "content": summary_prompt}]
    )

    st.markdown('<div class="triage-header">🩺 Triage Summary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="triage-box">{summary_result}</div>', unsafe_allow_html=True)

    # Detailed Report Button
    #if st.button("📋 View Full Detailed Report"):
        #st.session_state.show_detailed = True
        #st.session_state.summary_result = summary_result
        #st.rerun()


# ---------------- DOWNLOAD DETAILED REPORT DIRECTLY ----------------
# ---------------- DOWNLOAD DETAILED REPORT DIRECTLY ----------------

# IMPROVED PROMPT - More explicit formatting instructions
detailed_prompt = f"""
You are a medical triage assistant. Create a detailed clinical triage report with the following EXACT structure.

Use this format for each section:

Section Name:
Content here (use dashes - for bullet points)

Patient Information:
Name: {patient_name}
Age: {patient_age}
Sex: {patient_sex}

Based on this conversation:
{last_assistant_reply}

Now provide the following sections:

Risk Level:
[Provide risk assessment - Low, Moderate, or High with brief explanation]

Key Symptoms:
- [List main symptoms with dashes]
- [One symptom per line]

Chief Complaint:
[Brief description of main presenting issue]

History of Present Illness:
[Detailed narrative of the patient's condition]

Home Care Advice:
- [Provide specific home care recommendations]
- [Use dashes for each point]

OTC Guidance:
- [Over-the-counter medication suggestions if appropriate]
- [Include precautions]

Monitoring Advice:
- [What symptoms to monitor]
- [When to seek further care]

Health Checks:
[Recommended medical evaluations or tests if needed]

Reassurance:
[Calm, supportive message to patient]

Safety Disclaimer:
[Standard medical disclaimer about seeking professional care]

IMPORTANT: 
- Use simple text, NO markdown symbols like ** or #
- Use dashes (-) for bullet points
- Each section must start with section name followed by colon (:)
- Provide actual medical content, not placeholders
"""

detailed_result = generate_reply(
    model_choice,
    [{"role": "user", "content": detailed_prompt}]
)

# Check if we got a valid response
if not detailed_result or len(detailed_result.strip()) < 50:
    st.error("⚠️ The AI model returned an empty or very short response. Please try again.")
    st.write("**What was returned:**")
    st.code(detailed_result)
    st.stop()

# Optional: Display the response in the UI for debugging
#if st.checkbox("🔍 Show LLM Response (Debug)", value=False):
    #st.write("### AI Generated Content")
    #st.text_area("Full Response", detailed_result, height=300)
    #st.write("---")

# Generate PDF
pdf_path = f"Triage_Report_{session_id}.pdf"

doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                       topMargin=0.75*inch,
                       bottomMargin=0.75*inch,
                       leftMargin=0.75*inch,
                       rightMargin=0.75*inch)
elements = []
styles = getSampleStyleSheet()

# ---------------- CUSTOM STYLES ----------------
title_style = ParagraphStyle(
    name='CustomTitle',
    parent=styles['Heading1'],
    fontSize=24,
    textColor=colors.HexColor("#0B2E59"),
    spaceAfter=20,
    alignment=1,  # Center alignment
    fontName='Helvetica-Bold'
)

section_header_style = ParagraphStyle(
    name='SectionHeader',
    fontSize=12,
    textColor=colors.white,
    fontName='Helvetica-Bold',
    leftIndent=10,
    spaceAfter=0,
    spaceBefore=0
)

content_style = ParagraphStyle(
    name='ContentText',
    parent=styles['Normal'],
    fontSize=10,
    textColor=colors.black,
    leftIndent=0,
    spaceAfter=4,
    leading=14
)

bullet_style = ParagraphStyle(
    name='BulletText',
    parent=content_style,
    leftIndent=20,
    bulletIndent=10,
    spaceAfter=4
)

# ---------------- TITLE ----------------
elements.append(Paragraph("Clinical Triage Report", title_style))
elements.append(Spacer(1, 0.3 * inch))

# ---------------- PATIENT INFO BOX ----------------
patient_data = [
    [Paragraph("<b>Patient Name:</b>", content_style), Paragraph(patient_name, content_style)],
    [Paragraph("<b>Age / Sex:</b>", content_style), Paragraph(f"{patient_age} / {patient_sex}", content_style)],
    [Paragraph("<b>Report ID:</b>", content_style), Paragraph(session_id, content_style)],
]

patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
patient_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#E8EFF5")),
    ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor("#4A7BA7")),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 12),
    ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ('TOPPADDING', (0, 0), (-1, -1), 10),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
]))

elements.append(patient_table)
elements.append(Spacer(1, 0.4 * inch))

# ---------------- FORMAT SECTIONS CLEANLY ----------------
lines = detailed_result.split("\n")
sections = {}
current_section = None
current_content = []

for line in lines:
    line = line.strip()
    
    # Skip empty lines
    if not line:
        continue
    
    # Remove markdown formatting
    line = line.replace('**', '').replace('*', '').replace('#', '')
    
    # Check if this is a section header (ends with colon, not starting with dash)
    if ":" in line and not line.startswith("-") and not line.startswith("•"):
        # Save previous section if exists
        if current_section and current_content:
            sections[current_section] = current_content
        
        # Start new section
        current_section = line.replace(":", "").strip()
        current_content = []
    elif current_section:
        # Add content to current section
        current_content.append(line)

# Add the last section
if current_section and current_content:
    sections[current_section] = current_content

# Verify we have sections
if len(sections) == 0:
    st.error("⚠️ No sections were found in the AI response. The report cannot be generated.")
    st.write("**AI Response:**")
    st.text_area("Response", detailed_result, height=200)
    st.stop()

# ---------------- RENDER SECTIONS PROPERLY ----------------
for section_title, content_lines in sections.items():
    
    # Section Header Bar with blue background
    header_table = Table(
        [[Paragraph(section_title, section_header_style)]],
        colWidths=[6.5*inch]
    )
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#4A7BA7")),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(header_table)
    
    # White Content Box
    content_paragraphs = []
    for item in content_lines:
        if item.strip().startswith('-'):
            # Convert dash to bullet point
            text = item.strip()[1:].strip()
            content_paragraphs.append(Paragraph(f"• {text}", bullet_style))
        else:
            # Regular paragraph
            content_paragraphs.append(Paragraph(item, content_style))
    
    content_table = Table(
        [[content_paragraphs]],
        colWidths=[6.5*inch]
    )
    content_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#C0C0C0")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('RIGHTPADDING', (0, 0), (-1, -1), 15),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
    ]))
    elements.append(content_table)
    elements.append(Spacer(1, 0.25 * inch))

# Build the PDF
try:
    doc.build(elements)
    st.success(f"✅ PDF generated successfully with {len(sections)} sections!")
except Exception as e:
    st.error(f"❌ Error generating PDF: {str(e)}")
    st.stop()

# ---------------- DOWNLOAD BUTTON ----------------
with open(pdf_path, "rb") as f:
    st.download_button(
        label="📄 Download the Full Detailed Report",
        data=f,
        file_name="Clinical_Triage_Report.pdf",
        mime="application/pdf"
    )
