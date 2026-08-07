"""
AI Meeting Assistant
=====================
Upload meeting audio or notes -> get a summary, extracted & assigned action
items, a draft follow-up email, automatic Minutes of Meeting (PDF),
multilingual (English/Hindi/Hinglish) output, a chat interface to ask
questions about a meeting, and a team performance dashboard, plus a host/join meeting room.

Install dependencies:
    pip install -r requirements.txt

Run:
    streamlit run app.py

You will need a Google Gemini API key (https://aistudio.google.com/apikey).

NOTE ON DEPLOYMENT:
This app must run on Python 3.11 or 3.12. Python 3.14 breaks the compiled
protobuf C extension that streamlit/google-generativeai depend on
(TypeError: Metaclasses with custom tp_new are not supported), so make sure
your host picks up runtime.txt / .python-version and requirements.txt
alongside this file.

NOTE ON PDF FONT:
The PDF export embeds a bundled Unicode font (Noto Sans Devanagari) instead
of the built-in Latin-only Helvetica, so English, Hindi (Devanagari), and
Hinglish all render correctly in the MoM PDF. The .ttf files must be present
under fonts/ (see FONT_REGULAR / FONT_BOLD below) — see the README section
of this file for where to get them.
"""

# =========================== STEP 1: LOAD MODULES ===========================
import os
import json
import re
import time
import uuid
from datetime import datetime, date

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from fpdf import FPDF

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# =========================== STEP 2: APP CONFIG =============================

st.set_page_config(
    page_title="MeetFlow — AI Meeting Assistant",
    page_icon="meetflow_logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================== STEP 3A: PREMIUM UI ==============================

st.markdown("""

<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --mf-bg: #020617;
    --mf-blue: #2563eb;
    --mf-cyan: #06d6f5;
    --mf-card: #050f26;
    --mf-border: rgba(96,165,250,.24);
    --mf-text: #f8fafc;
    --mf-muted: #94a3b8;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    min-height: 100vh;
    background:
        radial-gradient(circle at 88% 2%, rgba(6,214,245,.10), transparent 24%),
        radial-gradient(circle at 40% 0%, rgba(37,99,235,.11), transparent 28%),
        linear-gradient(135deg, #020617 0%, #030916 48%, #07152f 100%);
    color: var(--mf-text);
}

.stApp header, [data-testid="stHeader"] {
    background: rgba(2,6,23,.86) !important;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.35rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 50% 0%, rgba(37,99,235,.12), transparent 25%),
        linear-gradient(180deg, #020817 0%, #03112b 100%) !important;
    border-right: 1px solid rgba(59,130,246,.16);
}

section[data-testid="stSidebar"] > div { background: transparent !important; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] label {
    color: #bfdbfe !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #07152f !important;
    color: #f8fafc !important;
    border: 1px solid rgba(96,165,250,.28) !important;
    border-radius: 12px !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(148,163,184,.15) !important;
}

.mf-topbar {
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    min-height: 135px;
    gap: 22px;
    padding: 22px 30px;
    margin-bottom: 22px;
    border: 1px solid rgba(37,99,235,.38);
    border-radius: 20px;
    background:
        radial-gradient(ellipse at 72% 50%, rgba(14,165,255,.13), transparent 35%),
        linear-gradient(110deg, rgba(3,10,29,.98), rgba(4,16,43,.88));
    box-shadow: 0 18px 60px rgba(0,0,0,.28), inset 0 1px rgba(255,255,255,.035);
}

.mf-topbar::after {
    content: "";
    position: absolute;
    right: -8%;
    top: 15%;
    width: 55%;
    height: 80%;
    background: repeating-radial-gradient(
        ellipse at center,
        transparent 0 10px,
        rgba(6,214,245,.08) 11px 12px,
        transparent 13px 22px
    );
    opacity: .45;
    transform: rotate(-7deg);
    pointer-events: none;
}

.mf-brand {
    font-size: 36px;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #fff;
    margin: 0;
    position: relative;
    z-index: 2;
}
.mf-brand span { color: #06d6f5; }
.mf-subtitle {
    color: #b9c9df;
    font-size: 14px;
    margin-top: 6px;
    position: relative;
    z-index: 2;
}
.mf-badge {
    margin-left: auto;
    position: relative;
    z-index: 3;
    padding: 9px 15px;
    border-radius: 999px;
    background: rgba(14,165,255,.08);
    color: #38bdf8;
    font-size: 12px;
    font-weight: 800;
    border: 1px solid rgba(14,165,255,.38);
    white-space: nowrap;
}

div[data-testid="stTabs"] > div:first-child {
    gap: 4px;
    background: rgba(2,8,23,.74);
    padding: 5px;
    border-radius: 15px;
    border: 1px solid rgba(96,165,250,.18);
    box-shadow: 0 10px 30px rgba(0,0,0,.20);
}
div[data-testid="stTabs"] button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    background: transparent !important;
}
div[data-testid="stTabs"] button:hover {
    color: #e0f2fe !important;
    background: rgba(37,99,235,.10) !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #fff !important;
    background: linear-gradient(135deg,#0b2d73,#1254d8) !important;
    box-shadow: 0 0 20px rgba(37,99,235,.22);
    border-bottom: 2px solid #06d6f5 !important;
}

div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button {
    border-radius: 11px !important;
    border: 1px solid rgba(96,165,250,.24) !important;
    background: #07152f !important;
    color: #e0f2fe !important;
    font-weight: 700 !important;
    min-height: 42px;
    box-shadow: 0 5px 18px rgba(0,0,0,.18);
}
div[data-testid="stButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:hover {
    border-color: #06b6d4 !important;
    color: #fff !important;
    box-shadow: 0 0 22px rgba(6,182,212,.15);
    transform: translateY(-1px);
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg,#0b4cc9,#087eea) !important;
    color: #fff !important;
    border: 1px solid #1998ff !important;
    box-shadow: 0 8px 28px rgba(14,116,255,.25);
}

div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-testid="stDateInput"] input {
    background: #050f26 !important;
    color: #f8fafc !important;
    border: 1px solid rgba(96,165,250,.28) !important;
    border-radius: 11px !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus,
div[data-testid="stDateInput"] input:focus {
    border-color: #22d3ee !important;
    box-shadow: 0 0 0 1px #22d3ee, 0 0 20px rgba(34,211,238,.08) !important;
}

div[data-testid="stTextInput"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stDateInput"] label,
div[data-testid="stFileUploader"] label,
div[data-testid="stRadio"] label,
div[data-testid="stSelectbox"] label {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
}

div[data-testid="stRadio"] label p { color: #e2e8f0 !important; }

div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #050f26 !important;
    color: #f8fafc !important;
    border: 1px solid rgba(96,165,250,.28) !important;
    border-radius: 11px !important;
}

div[data-testid="stFileUploader"] section {
    background: #050f26 !important;
    border: 1px dashed rgba(96,165,250,.34) !important;
    border-radius: 14px !important;
}
div[data-testid="stFileUploader"] section * { color: #cbd5e1 !important; }

div[data-testid="stMetric"] {
    background: linear-gradient(145deg,#07152f,#030b1e) !important;
    border: 1px solid rgba(96,165,250,.20) !important;
    border-radius: 15px;
    padding: 15px 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,.22);
}
div[data-testid="stMetric"] label { color: #94a3b8 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f8fafc !important; }

div[data-testid="stExpander"] {
    border: 1px solid rgba(96,165,250,.20) !important;
    border-radius: 14px !important;
    background: rgba(4,13,32,.72) !important;
}
div[data-testid="stExpander"] summary { color: #e2e8f0 !important; }

div[data-testid="stDataFrame"] {
    border: 1px solid rgba(96,165,250,.18);
    border-radius: 12px;
    overflow: hidden;
}

.stMarkdown, .stCaption, p, li { color: #cbd5e1; }
.stAlert {
    background: #07152f !important;
    border: 1px solid rgba(96,165,250,.22) !important;
    color: #dbeafe !important;
}

.mf-section {
    margin: 20px 0 9px;
    font-size: 25px;
    font-weight: 800;
    color: #f8fafc;
}
.mf-kicker {
    color: #38bdf8;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.4px;
}
.mf-footer {
    margin-top: 35px;
    padding: 18px;
    text-align: center;
    color: #64748b;
    font-size: 12px;
}
.mf-side-logo {
    text-align: center;
    padding: 7px 0 18px;
}
.mf-side-logo img {
    width: 175px;
    max-width: 100%;
    height: auto;
    filter: drop-shadow(0 0 16px rgba(6,214,245,.13));
}
.mf-side-kicker {
    color: #60a5fa;
    font-size: 11px;
    letter-spacing: 1.6px;
    font-weight: 800;
    margin-top: 8px;
}
::selection { background: #075985; color: white; }
</style>

""", unsafe_allow_html=True)

# Embed the MeetFlow logo directly into the app so it also works on Streamlit Cloud.
MEETFLOW_LOGO_B64 = "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"
MEETFLOW_LOGO_DATA = "data:image/png;base64," + MEETFLOW_LOGO_B64

st.markdown(
    f"""
    <div class="mf-topbar">
        <div style="display:flex;align-items:center;gap:18px;position:relative;z-index:2;">
            <img src="{MEETFLOW_LOGO_DATA}" style="width:118px;height:auto;max-height:72px;object-fit:contain;">
            <div>
                <div class="mf-brand">Meet<span>Flow</span></div>
                <div class="mf-subtitle">AI-powered meeting intelligence • capture, understand, act</div>
            </div>
        </div>
        <div class="mf-badge">✦ AI WORKSPACE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    f"""
    <div class="mf-side-logo">
        <img src="{MEETFLOW_LOGO_DATA}">
        <div class="mf-side-kicker">MEETING INTELLIGENCE</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.title("⚙️ Workspace Settings")


HISTORY_FILE = "meeting_history.json"
ROOMS_FILE = "meeting_rooms.json"
AUDIO_DIR = "meeting_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

AUDIO_TYPES = ["mp3", "wav", "m4a", "aac", "ogg", "flac"]
TEXT_TYPES = ["txt", "md"]

# Unicode font for the MoM PDF (needed for Devanagari/Hindi + Hinglish + English).
# fpdf2's built-in "Helvetica" is Latin-1 only, so we embed Noto Sans Devanagari,
# which covers Devanagari, Latin, and most punctuation in one font.
# Download the two .ttf files (free, OFL license) and place them here:
#   https://fonts.google.com/noto/specimen/Noto+Sans+Devanagari
FONT_DIR = "fonts"
FONT_REGULAR = os.path.join(FONT_DIR, "NotoSansDevanagari-Regular.ttf")
FONT_BOLD = os.path.join(FONT_DIR, "NotoSansDevanagari-Bold.ttf")
UNICODE_FONT_AVAILABLE = os.path.exists(FONT_REGULAR) and os.path.exists(FONT_BOLD)

LANGUAGE_INSTRUCTIONS = {
    "English": "Respond in clear, professional English.",
    "Hindi": "Respond in Hindi, written in Devanagari script.",
    "Hinglish": (
        "Respond in Hinglish — a natural mix of Hindi and English written in "
        "Roman/Latin script, the way it's commonly written in Indian workplace "
        "chats and emails."
    ),
}

# =========================== STEP 3: SIDEBAR CONFIG =========================

st.sidebar.markdown(
    '<div style="text-align:center;padding:10px 0 18px;">'
    '<img src="meetflow_logo.png" width="170" style="max-width:100%;height:auto;">'
    '<div style="font-size:12px;letter-spacing:1.5px;color:#9fb5d8;font-weight:700;margin-top:8px;">MEETING INTELLIGENCE</div>'
    '</div>',
    unsafe_allow_html=True,
)
st.sidebar.title("⚙️ Workspace Settings")

GOOGLE_API_KEY = st.sidebar.text_input("GOOGLE_API_KEY", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    genai.configure(api_key=GOOGLE_API_KEY)

model_choice = st.sidebar.selectbox(
    "Model",
    ["gemini-3.6-flash", "gemini-3.5-flash-lite"],
    index=0,
    help="gemini-2.0-flash and gemini-1.5-flash have been retired by Google. "
    "gemini-3.6-flash is the current general-purpose GA model; "
    "gemini-3.5-flash-lite is faster/cheaper for high-volume use.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Language")
output_language = st.sidebar.selectbox(
    "Output language (summary,chat)",
    ["English", "Hindi", "Hinglish"],
    index=0,
    help="Meeting audio can be spoken in Hindi/English/Hinglish regardless of this setting — "
    "this only controls the language of the generated summary/email/MoM/chat answers.",
)
language_instruction = LANGUAGE_INSTRUCTIONS[output_language]

st.sidebar.markdown("---")
team_members_raw = st.sidebar.text_area(
    "Team members (comma-separated)",
    placeholder="Alice, Bob, Charlie",
    help="Used so the assistant can assign action items to real people when it can infer an owner.",
)
team_members = [m.strip() for m in team_members_raw.split(",") if m.strip()]

st.sidebar.markdown("---")
st.sidebar.caption("History is saved locally to meeting_history.json")


# =========================== STEP 4: HISTORY STORAGE ========================

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def add_meeting_record(record):
    history = load_history()
    history.insert(0, record)
    save_history(history)


def load_rooms():
    if os.path.exists(ROOMS_FILE):
        try:
            with open(ROOMS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_rooms(rooms):
    with open(ROOMS_FILE, "w", encoding="utf-8") as f:
        json.dump(rooms, f, indent=2, ensure_ascii=False)


def create_meeting_room(title, host, participants, agenda):
    code = uuid.uuid4().hex[:6].upper()
    rooms = load_rooms()
    rooms[code] = {
        "code": code,
        "title": title,
        "host": host,
        "participants": participants,
        "agenda": agenda,
        "notes": "",
        "started_at": None,
        "ended_at": None,
    }
    save_rooms(rooms)
    return rooms[code]


def update_meeting_room(code, **updates):
    rooms = load_rooms()
    if code not in rooms:
        return None
    rooms[code].update(updates)
    save_rooms(rooms)
    return rooms[code]


def get_meeting_room(code):
    return load_rooms().get(code)


def update_action_item_status(record_id, item_index):
    """Toggle an action item's status between Pending and Completed, persisted to disk."""
    history = load_history()
    for h in history:
        if h["id"] == record_id:
            try:
                item = h["action_items"][item_index]
            except IndexError:
                return
            item["status"] = "Pending" if item.get("status") == "Completed" else "Completed"
            break
    save_history(history)


# =========================== STEP 5: LLM HELPERS =============================

def get_llm():
    return ChatGoogleGenerativeAI(model=model_choice, temperature=0.3)


def transcribe_audio(file_path: str) -> str:
    """Transcribe audio using Gemini's native audio understanding.

    Transcription preserves whatever language(s) were actually spoken
    (English / Hindi / Hinglish / code-switched) rather than translating,
    so downstream summarization can still work from the true source text.
    """
    audio_file = genai.upload_file(path=file_path)

    while audio_file.state.name == "PROCESSING":
        time.sleep(1)
        audio_file = genai.get_file(audio_file.name)

    if audio_file.state.name == "FAILED":
        raise RuntimeError("Audio processing failed on Google's servers.")

    model = genai.GenerativeModel(model_choice)
    response = model.generate_content(
        [
            "Transcribe this meeting recording as accurately as possible. "
            "The speakers may talk in English, Hindi, or Hinglish (code-switched "
            "Hindi-English) — transcribe faithfully in whatever language/script was "
            "actually spoken; do not translate. Label speakers as Speaker 1, Speaker 2, "
            "etc. if they are distinguishable. Return only the transcript text.",
            audio_file,
        ]
    )
    return response.text


def extract_json_block(text: str):
    """Pull the first JSON array/object out of a model response."""
    candidates = []

    fenced = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))

    array_match = re.search(r"\[.*?\]", text, re.DOTALL)
    if array_match:
        candidates.append(array_match.group(0))

    obj_match = re.search(r"\{.*?\}", text, re.DOTALL)
    if obj_match:
        candidates.append(obj_match.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError("No valid JSON found in model output.")


def summarize_meeting(llm, transcript: str, language_instruction: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        """You are an expert meeting assistant. Summarize the following meeting
transcript/notes into a concise, well-organized summary using short paragraphs
or bullet points. Cover: purpose of the meeting, key discussion points,
decisions made, and open questions. Do not invent details not present in the text.
{language_instruction}

Meeting content:
{content}

Summary:"""
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"content": transcript, "language_instruction": language_instruction})


def extract_action_items(llm, transcript: str, team_members: list, language_instruction: str) -> list:
    team_hint = (
        f"Known team members: {', '.join(team_members)}. Assign items to one of "
        f"these names ONLY if the transcript clearly indicates who owns the task; "
        f"otherwise use \"Unassigned\"."
        if team_members
        else 'No team member list was provided, so set "owner" to the name mentioned '
        'in the transcript, or "Unassigned" if unclear.'
    )

    prompt = ChatPromptTemplate.from_template(
        """Extract every action item / task from the following meeting content.
{team_hint}
Keep "task", "owner", and "deadline" values in plain text. {language_instruction}

Return ONLY valid JSON: a list of objects with keys "task", "owner", and "deadline"
("deadline" should be a date/timeframe if mentioned, otherwise "TBD"). No markdown
fences, no commentary — JSON only. If there are no action items, return [].

Meeting content:
{content}

JSON:"""
    )
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({
        "content": transcript,
        "team_hint": team_hint,
        "language_instruction": language_instruction,
    })
    try:
        items = extract_json_block(raw)
        if isinstance(items, dict):
            items = [items]
        cleaned = []
        for item in items:
            if isinstance(item, dict):
                item.setdefault("status", "Pending")
                cleaned.append(item)
            else:
                cleaned.append({"task": str(item), "owner": "Unassigned", "deadline": "TBD", "status": "Pending"})
        return cleaned
    except (ValueError, json.JSONDecodeError):
        return [{"task": raw.strip(), "owner": "Unassigned", "deadline": "TBD", "status": "Pending"}]


def draft_followup_email(llm, summary: str, action_items: list, title: str, language_instruction: str) -> str:
    items_text = "\n".join(
        f"- {i.get('task', '')} (Owner: {i.get('owner', 'Unassigned')}, Due: {i.get('deadline', 'TBD')})"
        for i in action_items
    ) or "- No specific action items were identified."

    prompt = ChatPromptTemplate.from_template(
        """Write a professional, concise follow-up email to meeting attendees.
{language_instruction}

Meeting title: {title}

Summary:
{summary}

Action items:
{items}

The email should include: a brief thank-you/recap line, the key summary points
as short bullets, a clearly formatted action items section (task, owner, due date),
and a friendly closing line. Include a subject line at the top formatted as
"Subject: ...". Return only the email text."""
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "title": title, "summary": summary, "items": items_text,
        "language_instruction": language_instruction,
    })


def answer_meeting_question(llm, transcript: str, chat_history: list, question: str, language_instruction: str) -> str:
    """Answer a question about a specific meeting, grounded only in its transcript."""
    history_text = "\n".join(f"{role}: {msg}" for role, msg in chat_history[-6:]) or "(no prior turns)"
    prompt = ChatPromptTemplate.from_template(
        """You are answering questions about ONE specific meeting, using ONLY the
meeting content below as your source of truth. If the answer isn't in the
content, say plainly that it wasn't mentioned in the meeting — never invent
details. {language_instruction}

Meeting content:
{content}

Conversation so far:
{history}

Question: {question}

Answer:"""
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "content": transcript, "history": history_text,
        "question": question, "language_instruction": language_instruction,
    })


def _sanitize_for_pdf(text: str, max_token_len: int = 55) -> str:
    """Prepare text for the fixed-width PDF renderer.

    Two things break fpdf2's line-wrapping and must be handled before
    rendering: (1) raw markdown syntax from the LLM's output (**bold**,
    [text](url), etc.) showing up as literal symbols, and (2) any single
    unbroken run of characters (a long URL, a run-on compound word) that's
    wider than the page — fpdf2 raises "Not enough horizontal space to
    render a single character" when it can't find a place to break a word.
    We strip markdown formatting and insert breakable spaces into any
    overly long token so every line can always be wrapped.
    """
    if not text:
        return ""

    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)          # **bold**
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)  # *italic*
    text = re.sub(r"__(.*?)__", r"\1", text)               # __bold__
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)  # # headings
    text = re.sub(r"`([^`]*)`", r"\1", text)               # `code`
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)  # [text](url)

    def _break_long_token(match):
        token = match.group(0)
        return " ".join(token[i:i + max_token_len] for i in range(0, len(token), max_token_len))

    text = re.sub(r"\S{%d,}" % (max_token_len + 1), _break_long_token, text)

    if UNICODE_FONT_AVAILABLE:
        # Embedded Unicode font handles Devanagari, Latin, and most
        # punctuation natively — no lossy re-encoding needed.
        return text
    # Fallback path only: no embedded font found, so we're stuck on the
    # built-in Latin-1 Helvetica and must drop unsupported characters
    # (this is what causes blank/missing glyphs for Hindi text).
    return text.encode("latin-1", "replace").decode("latin-1")


def generate_mom_pdf(meeting_title: str, meeting_date: str, summary: str, action_items: list) -> bytes:
    """Minutes of Meeting as a PDF.

    Uses the bundled Noto Sans Devanagari Unicode font when available, so
    English, Hindi (Devanagari), and Hinglish content all render correctly.
    If the font files aren't present under fonts/, falls back to the
    built-in Latin-1 Helvetica (Devanagari text will show blank/missing
    glyphs in that fallback case).
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    w = pdf.epw  # effective page width (explicit, safer than w=0's auto-calc)

    if UNICODE_FONT_AVAILABLE:
        pdf.add_font("NotoSans", "", FONT_REGULAR)
        pdf.add_font("NotoSans", "B", FONT_BOLD)
        font_name = "NotoSans"
    else:
        font_name = "Helvetica"

    pdf.set_font(font_name, "B", 16)
    pdf.multi_cell(w, 10, _sanitize_for_pdf("Minutes of Meeting"), align="C")
    pdf.ln(2)

    pdf.set_font(font_name, "", 11)
    pdf.multi_cell(w, 7, _sanitize_for_pdf(f"Meeting: {meeting_title}"))
    pdf.multi_cell(w, 7, _sanitize_for_pdf(f"Date: {meeting_date}"))
    pdf.ln(4)

    pdf.set_font(font_name, "B", 13)
    pdf.multi_cell(w, 8, _sanitize_for_pdf("Summary"))
    pdf.set_font(font_name, "", 11)
    pdf.multi_cell(w, 6, _sanitize_for_pdf(summary))
    pdf.ln(4)

    pdf.set_font(font_name, "B", 13)
    pdf.multi_cell(w, 8, _sanitize_for_pdf("Action Items"))
    pdf.set_font(font_name, "", 11)
    if action_items:
        for i, item in enumerate(action_items, 1):
            line = (
                f"{i}. {item.get('task', '')} | Owner: {item.get('owner', 'Unassigned')} | "
                f"Due: {item.get('deadline', 'TBD')} | Status: {item.get('status', 'Pending')}"
            )
            pdf.multi_cell(w, 6, _sanitize_for_pdf(line))
    else:
        pdf.multi_cell(w, 6, _sanitize_for_pdf("No action items identified."))

    output = pdf.output(dest="S")
    return bytes(output)


def build_dashboard_dataframe(history: list) -> pd.DataFrame:
    """Flatten every action item across all saved meetings into one table
    for the Team Performance Dashboard."""
    rows = []
    for record in history:
        for item in record.get("action_items", []):
            rows.append({
                "Meeting": record.get("title", ""),
                "Date": record.get("created_at", "")[:10],
                "Owner": item.get("owner", "Unassigned") or "Unassigned",
                "Task": item.get("task", ""),
                "Deadline": item.get("deadline", "TBD"),
                "Status": item.get("status", "Pending"),
            })
    return pd.DataFrame(rows, columns=["Meeting", "Date", "Owner", "Task", "Deadline", "Status"])


# =========================== STEP 6: UI — TABS ===============================

tab_host, tab_new, tab_chat, tab_history, tab_dashboard = st.tabs(
    ["🎥  Host", "📝  New Meeting", "💬  AI Chat", "📚  History", "📈  Dashboard"]
)

# ------------------------------- HOST MEETING TAB ------------------------------
with tab_host:
    st.markdown('<div class="mf-kicker">LIVE COLLABORATION</div><div class="mf-section">🎥 Host an AI Meeting</div>', unsafe_allow_html=True)
    st.caption("Create a meeting room, share the 6-character code, capture live notes, and let the AI generate the final meeting outputs when you end the meeting.")

    host_mode = st.radio(
        "Meeting room",
        ["Create a new meeting", "Join an existing meeting"],
        horizontal=True,
        key="host_mode",
    )

    if host_mode == "Create a new meeting":
        host_title = st.text_input("Meeting title", value=f"Live Meeting - {datetime.now().strftime('%Y-%m-%d')}", key="host_title")
        host_name = st.text_input("Host name", placeholder="Your name", key="host_name")
        host_participants_raw = st.text_input("Participants (comma-separated)", placeholder="Alice, Rahul, Priya", key="host_participants")
        host_agenda = st.text_area("Agenda", placeholder="1. Project progress\n2. Budget\n3. Next steps", height=120, key="host_agenda")

        if st.button("🚀 Create Meeting Room", type="primary", key="create_room"):
            participants = [x.strip() for x in host_participants_raw.split(",") if x.strip()]
            room = create_meeting_room(host_title, host_name or "Host", participants, host_agenda)
            st.session_state["active_room_code"] = room["code"]
            st.session_state["room_running"] = False
            st.success("Meeting room created!")

    active_code = st.session_state.get("active_room_code")
    if active_code:
        room = get_meeting_room(active_code)
        if room:
            st.markdown("### 🔗 Your Meeting Room")
            c1, c2, c3 = st.columns(3)
            c1.metric("Room Code", room["code"])
            c2.metric("Host", room["host"])
            c3.metric("Participants", len(room.get("participants", [])))

            st.info(f"Share this code with participants: **{room['code']}**")

            if room.get("agenda"):
                with st.expander("📋 Meeting Agenda", expanded=True):
                    st.write(room["agenda"])

            if not room.get("started_at"):
                if st.button("🔴 Start Meeting", type="primary", key="start_room"):
                    update_meeting_room(active_code, started_at=datetime.now().isoformat(timespec="seconds"))
                    st.session_state["room_running"] = True
                    st.rerun()
            else:
                st.success(f"Meeting started at {room['started_at']}")

                participants_text = ", ".join(room.get("participants", [])) or "No participants added"
                st.markdown(f"**👥 Participants:** {participants_text}")

                live_notes = st.text_area(
                    "📝 Live meeting notes",
                    value=room.get("notes", ""),
                    height=220,
                    placeholder="Type important discussion points, decisions, and tasks while the meeting is happening...",
                    key=f"live_notes_{active_code}",
                )

                if st.button("💾 Save Live Notes", key="save_live_notes"):
                    update_meeting_room(active_code, notes=live_notes)
                    st.success("Live notes saved.")

                if not room.get("ended_at"):
                    if st.button("🛑 End Meeting & Generate AI Minutes", type="primary", key="end_room"):
                        notes_to_process = live_notes.strip()
                        if not notes_to_process:
                            st.warning("Add some meeting notes before ending the meeting so the AI has content to analyze.")
                        elif not GOOGLE_API_KEY:
                            st.warning("Enter your Google API key in the sidebar first.")
                        else:
                            try:
                                update_meeting_room(
                                    active_code,
                                    notes=notes_to_process,
                                    ended_at=datetime.now().isoformat(timespec="seconds"),
                                )
                                llm = get_llm()
                                with st.spinner("🤖 AI is preparing your meeting minutes..."):
                                    room_summary = summarize_meeting(llm, notes_to_process, language_instruction)
                                    room_actions = extract_action_items(llm, notes_to_process, team_members, language_instruction)
                                    room_email = draft_followup_email(llm, room_summary, room_actions, room["title"], language_instruction)

                                mom_date = datetime.now().strftime("%Y-%m-%d")
                                record = {
                                    "id": uuid.uuid4().hex,
                                    "title": room["title"],
                                    "created_at": datetime.now().isoformat(timespec="seconds"),
                                    "meeting_date": mom_date,
                                    "transcript": notes_to_process,
                                    "summary": room_summary,
                                    "action_items": room_actions,
                                    "email_draft": room_email,
                                }
                                add_meeting_record(record)
                                st.session_state["last_meeting_id"] = record["id"]
                                st.session_state["host_result"] = {
                                    "title": room["title"],
                                    "date": mom_date,
                                    "summary": room_summary,
                                    "actions": room_actions,
                                    "email": room_email,
                                }
                                st.success("🎉 Meeting ended and AI minutes were generated!")
                            except Exception as e:
                                st.error(f"Could not generate meeting minutes: {e}")

            host_result = st.session_state.get("host_result")
            if host_result:
                st.markdown("---")
                st.subheader("🤖 AI Meeting Results")
                st.markdown(host_result["summary"])
                st.markdown("**✅ Action Items**")
                if host_result["actions"]:
                    st.table([
                        {
                            "Task": i.get("task", ""),
                            "Owner": i.get("owner", "Unassigned"),
                            "Deadline": i.get("deadline", "TBD"),
                            "Status": i.get("status", "Pending"),
                        }
                        for i in host_result["actions"]
                    ])
                st.markdown("**✉️ Follow-up Email**")
                st.text_area("Email", value=host_result["email"], height=220, key="host_email_result")
                try:
                    host_pdf = generate_mom_pdf(host_result["title"], host_result["date"], host_result["summary"], host_result["actions"])
                    st.download_button(
                        "📄 Download Meeting MoM PDF",
                        data=host_pdf,
                        file_name=f"{host_result['title'].replace(' ', '_')}_MoM.pdf",
                        mime="application/pdf",
                        key="host_pdf_download",
                    )
                except Exception as pdf_err:
                    st.warning(f"PDF unavailable: {pdf_err}")

    if host_mode == "Join an existing meeting":
        join_code = st.text_input("Enter 6-character meeting code", max_chars=6, key="join_code").strip().upper()
        if st.button("🔗 Join Meeting", key="join_room"):
            room = get_meeting_room(join_code)
            if room:
                st.session_state["active_room_code"] = join_code
                st.success(f"Joined **{room['title']}** hosted by **{room['host']}**")
                st.rerun()
            else:
                st.error("Meeting room not found. Check the code and try again.")

    st.caption("Demo note: this is a Streamlit meeting-room layer with a shared room code and AI minutes workflow. It does not replace Zoom/Google Meet video infrastructure.")


# ------------------------------- NEW MEETING TAB ------------------------------
with tab_new:
    if not GOOGLE_API_KEY:
        st.warning("Enter your Google API key in the sidebar to get started.")

    meeting_title = st.text_input("Meeting title", value=f"Meeting - {datetime.now().strftime('%Y-%m-%d')}")
    meeting_date_input = st.date_input("Meeting date", value=date.today())

    input_mode = st.radio(
        "Input type",
        ["Upload audio recording", "Upload text notes", "Paste notes"],
        horizontal=True,
    )

    transcript_text = None
    audio_path = None

    if input_mode == "Upload audio recording":
        audio_file = st.file_uploader("Upload meeting audio", type=AUDIO_TYPES)
        if audio_file is not None:
            audio_path = os.path.join(AUDIO_DIR, f"{uuid.uuid4().hex}_{audio_file.name}")
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())
            st.audio(audio_file)

    elif input_mode == "Upload text notes":
        text_file = st.file_uploader("Upload meeting notes", type=TEXT_TYPES)
        if text_file is not None:
            transcript_text = text_file.read().decode("utf-8", errors="ignore")
            with st.expander("Preview uploaded notes"):
                st.text(transcript_text[:3000])

    else:  # Paste notes
        transcript_text = st.text_area("Paste meeting notes / transcript here", height=250)

    process_clicked = st.button("🚀 Process Meeting", type="primary", disabled=not GOOGLE_API_KEY)

    if process_clicked:
        if audio_path is None and not transcript_text:
            st.error("Please provide audio or notes before processing.")
        else:
            try:
                llm = get_llm()

                if audio_path is not None:
                    with st.spinner("Transcribing audio..."):
                        transcript_text = transcribe_audio(audio_path)

                with st.spinner("Summarizing discussion..."):
                    summary = summarize_meeting(llm, transcript_text, language_instruction)

                with st.spinner("Extracting & assigning action items..."):
                    action_items = extract_action_items(llm, transcript_text, team_members, language_instruction)

                with st.spinner("Drafting follow-up email..."):
                    email_draft = draft_followup_email(llm, summary, action_items, meeting_title, language_instruction)

                mom_date_str = meeting_date_input.strftime("%Y-%m-%d")

                # Save to history
                record = {
                    "id": uuid.uuid4().hex,
                    "title": meeting_title,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "meeting_date": mom_date_str,
                    "transcript": transcript_text,
                    "summary": summary,
                    "action_items": action_items,
                    "email_draft": email_draft,
                }
                add_meeting_record(record)
                st.session_state["last_meeting_id"] = record["id"]

                # Stash results in session_state so they survive reruns triggered
                # by clicking a download button — otherwise Streamlit's rerun
                # would wipe this block and force reprocessing every time.
                st.session_state["current_result"] = {
                    "title": meeting_title,
                    "transcript": transcript_text,
                    "summary": summary,
                    "action_items": action_items,
                    "email_draft": email_draft,
                    "mom_date_str": mom_date_str,
                }
                st.toast("Saved to meeting history ✅")

            except Exception as e:
                st.error(f"Something went wrong: {e}")

    # Render the most recently processed meeting's results. Reading from
    # session_state (instead of living inside "if process_clicked:") means
    # downloading the email, then the MoM PDF all just work — no need to
    # hit "Process Meeting" again between downloads.
    result = st.session_state.get("current_result")
    if result:
        st.success("Meeting processed!")

        with st.expander("📄 Transcript / Notes used", expanded=False):
            st.text(result["transcript"])

        st.subheader("📋 Summary")
        st.markdown(result["summary"])

        st.subheader("✅ Action Items")
        if result["action_items"]:
            st.table(
                [
                    {
                        "Task": i.get("task", ""),
                        "Owner": i.get("owner", "Unassigned"),
                        "Deadline": i.get("deadline", "TBD"),
                        "Status": i.get("status", "Pending"),
                    }
                    for i in result["action_items"]
                ]
            )
        else:
            st.info("No action items were identified.")

        st.subheader("✉️ Follow-up Email Draft")
        st.text_area("Email draft", value=result["email_draft"], height=300, key="email_draft_display")
        st.download_button(
            "Download email (.txt)",
            data=result["email_draft"],
            file_name=f"{result['title'].replace(' ', '_')}_followup.txt",
            key="dl_email_new",
        )

        st.subheader("📄 Minutes of Meeting")
        try:
            mom_pdf_bytes = generate_mom_pdf(result["title"], result["mom_date_str"], result["summary"], result["action_items"])
            st.download_button(
                "📄 Download MoM (PDF)",
                data=mom_pdf_bytes,
                file_name=f"{result['title'].replace(' ', '_')}_MoM.pdf",
                mime="application/pdf",
                key="dl_pdf_new",
            )
            if not UNICODE_FONT_AVAILABLE:
                st.caption("⚠️ Unicode font not found under fonts/ — Devanagari text won't render. See README note in app.py.")
        except Exception as pdf_err:
            st.error(f"Couldn't generate PDF: {pdf_err}")

# ------------------------------- ASK YOUR MEETING TAB --------------------------
with tab_chat:
    st.markdown('<div class="mf-kicker">MEETING INTELLIGENCE</div><div class="mf-section">💬 Ask Your Meeting</div>', unsafe_allow_html=True)
    st.caption('Example: "What did Rahul agree to do?" — answers are grounded only in that meeting\'s content.')

    history_for_chat = load_history()

    if not history_for_chat:
        st.info("Process a meeting first (New Meeting tab) — chat needs a meeting to reference.")
    elif not GOOGLE_API_KEY:
        st.warning("Enter your Google API key in the sidebar to chat with a meeting.")
    else:
        titles = [f"{h['title']} — {h['created_at'][:10]}" for h in history_for_chat]
        default_idx = 0
        last_id = st.session_state.get("last_meeting_id")
        if last_id:
            ids = [h["id"] for h in history_for_chat]
            if last_id in ids:
                default_idx = ids.index(last_id)

        selected_idx = st.selectbox("Meeting to chat with", range(len(titles)), format_func=lambda i: titles[i], index=default_idx)
        selected_meeting = history_for_chat[selected_idx]

        chat_key = f"chat_{selected_meeting['id']}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []  # list of (role, content)

        for role, content in st.session_state[chat_key]:
            with st.chat_message(role):
                st.markdown(content)

        question = st.chat_input(f"Ask something about \"{selected_meeting['title']}\"...")
        if question:
            st.session_state[chat_key].append(("user", question))
            with st.chat_message("user"):
                st.markdown(question)

            try:
                llm = get_llm()
                with st.spinner("Thinking..."):
                    answer = answer_meeting_question(
                        llm, selected_meeting["transcript"], st.session_state[chat_key],
                        question, language_instruction,
                    )
                st.session_state[chat_key].append(("assistant", answer))
                with st.chat_message("assistant"):
                    st.markdown(answer)
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# ------------------------------- HISTORY TAB ----------------------------------
with tab_history:
    history = load_history()

    if not history:
        st.info("No meetings saved yet. Process a meeting in the 'New Meeting' tab.")
    else:
        search = st.text_input("Search history by title")
        filtered = [h for h in history if search.lower() in h["title"].lower()] if search else history

        for record in filtered:
            with st.expander(f"🗂️ {record['title']}  —  {record['created_at']}"):
                st.markdown("**Summary**")
                st.markdown(record["summary"])

                st.markdown("**Action Items**")
                if record["action_items"]:
                    for idx, item in enumerate(record["action_items"]):
                        cols = st.columns([5, 2, 2, 2])
                        cols[0].write(item.get("task", ""))
                        cols[1].write(item.get("owner", "Unassigned"))
                        cols[2].write(item.get("deadline", "TBD"))
                        cols[3].checkbox(
                            "Done",
                            value=(item.get("status", "Pending") == "Completed"),
                            key=f"status_{record['id']}_{idx}",
                            on_change=update_action_item_status,
                            args=(record["id"], idx),
                        )
                else:
                    st.caption("No action items recorded.")

                st.markdown("**Follow-up Email**")
                st.text_area(
                    "Email",
                    value=record["email_draft"],
                    height=200,
                    key=f"email_{record['id']}",
                )

                st.markdown("**Minutes of Meeting**")
                mom_date_str = record.get("meeting_date", record["created_at"][:10])

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "Download email",
                        data=record["email_draft"],
                        file_name=f"{record['title'].replace(' ', '_')}_followup.txt",
                        key=f"dl_email_{record['id']}",
                    )
                with col2:
                    try:
                        mom_pdf_bytes = generate_mom_pdf(record["title"], mom_date_str, record["summary"], record["action_items"])
                        st.download_button(
                            "📄 MoM (PDF)",
                            data=mom_pdf_bytes,
                            file_name=f"{record['title'].replace(' ', '_')}_MoM.pdf",
                            mime="application/pdf",
                            key=f"dl_pdf_{record['id']}",
                        )
                    except Exception as pdf_err:
                        st.caption(f"PDF unavailable: {pdf_err}")

                if st.button("🗑️ Delete record", key=f"del_{record['id']}"):
                    history = [h for h in history if h["id"] != record["id"]]
                    save_history(history)
                    st.rerun()

# ------------------------------- TEAM DASHBOARD TAB -----------------------------
with tab_dashboard:
    st.markdown('<div class="mf-kicker">TEAM INSIGHTS</div><div class="mf-section">📈 Team Performance Dashboard</div>', unsafe_allow_html=True)
    st.caption("Completed vs. pending action items, tracked across all meetings. Mark items done in the History tab.")

    dash_history = load_history()
    df = build_dashboard_dataframe(dash_history)

    if df.empty:
        st.info("No action items yet. Process a meeting to populate the dashboard.")
    else:
        total = len(df)
        completed = int((df["Status"] == "Completed").sum())
        pending = total - completed

        m1, m2, m3 = st.columns(3)
        m1.metric("Total action items", total)
        m2.metric("Completed", completed)
        m3.metric("Pending", pending)

        st.markdown("**By team member**")
        pivot = (
            df.groupby(["Owner", "Status"]).size().unstack(fill_value=0)
        )
        for col in ["Completed", "Pending"]:
            if col not in pivot.columns:
                pivot[col] = 0
        pivot = pivot[["Completed", "Pending"]].sort_values("Pending", ascending=False)
        st.bar_chart(pivot)

        st.markdown("**All pending action items**")
        pending_df = df[df["Status"] == "Pending"].sort_values(["Owner", "Date"])
        if pending_df.empty:
            st.success("No pending action items — everything's done! 🎉")
        else:
            st.dataframe(pending_df[["Owner", "Task", "Meeting", "Deadline", "Date"]], use_container_width=True, hide_index=True)

        with st.expander("Full action item log"):
            st.dataframe(df, use_container_width=True, hide_index=True)

# =========================== MEETFLOW FOOTER ================================
st.markdown(
    '<div class="mf-footer">MeetFlow • AI Meeting Intelligence '
    '• Turn every conversation into clear next steps.</div>',
    unsafe_allow_html=True,
)
