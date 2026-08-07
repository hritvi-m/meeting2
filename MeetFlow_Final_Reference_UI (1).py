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
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================== STEP 3A: PREMIUM UI ==============================

st.markdown("""

<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --mf-bg: #020617;
    --mf-panel: #030a1c;
    --mf-panel2: #07152f;
    --mf-border: rgba(66, 153, 225, .25);
    --mf-blue: #1677ff;
    --mf-cyan: #06d6f5;
    --mf-white: #f8fbff;
    --mf-text: #dbe7f5;
    --mf-muted: #91a4bd;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    min-height: 100vh;
    background:
        radial-gradient(circle at 86% 4%, rgba(6,214,245,.08), transparent 24%),
        radial-gradient(circle at 44% 0%, rgba(37,99,235,.10), transparent 30%),
        linear-gradient(135deg, #020617 0%, #020817 48%, #06132c 100%);
    color: var(--mf-white);
}

.stApp header, [data-testid="stHeader"] {
    background: rgba(2,6,23,.88) !important;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.35rem;
    padding-bottom: 3rem;
}

/* ------------------------- SIDEBAR ------------------------- */
section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 45% 0%, rgba(37,99,235,.11), transparent 24%),
        linear-gradient(180deg, #020817 0%, #03112b 100%) !important;
    border-right: 1px solid rgba(59,130,246,.18);
}

section[data-testid="stSidebar"] > div { background: transparent !important; }
section[data-testid="stSidebar"] * { color: #e5eefb !important; }

section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #06142f !important;
    color: #f8fafc !important;
    border: 1px solid rgba(96,165,250,.28) !important;
    border-radius: 11px !important;
}

section[data-testid="stSidebar"] label {
    color: #bfd2eb !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(148,163,184,.14) !important;
}

.mf-side-brand {
    text-align: left;
    padding: 10px 4px 17px;
}
.mf-side-icon {
    width: 58px;
    height: 58px;
    object-fit: contain;
    vertical-align: middle;
    filter: drop-shadow(0 0 13px rgba(6,214,245,.18));
}
.mf-side-wordmark {
    display: inline-block;
    margin-left: 5px;
    vertical-align: middle;
    font-size: 25px;
    line-height: 1;
    font-weight: 800;
    letter-spacing: -1px;
    color: #ffffff;
}
.mf-side-wordmark span { color: #06d6f5; }
.mf-side-kicker {
    color: #60a5fa !important;
    font-size: 10px;
    letter-spacing: 1.5px;
    font-weight: 800;
    margin: 8px 0 0 9px;
}
.mf-side-nav {
    display: flex;
    flex-direction: column;
    gap: 7px;
    margin: 5px 0 18px;
}
.mf-side-nav-item {
    padding: 10px 12px;
    border-radius: 10px;
    color: #d9e6f7 !important;
    font-size: 14px;
    font-weight: 600;
    border: 1px solid transparent;
}
.mf-side-nav-item.active {
    background: linear-gradient(135deg, rgba(11,53,130,.85), rgba(18,84,216,.58));
    border: 1px solid rgba(37,140,255,.55);
    box-shadow: 0 0 20px rgba(37,99,235,.12);
    color: #ffffff !important;
}
.mf-side-divider {
    height: 1px;
    background: rgba(148,163,184,.15);
    margin: 3px 0 19px;
}
.mf-side-config-title {
    color: #f8fafc !important;
    font-size: 20px;
    font-weight: 800;
    margin: 0 0 15px;
}
.mf-side-section {
    color: #60a5fa !important;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.5px;
    margin: 0 0 11px;
}
.mf-side-info {
    margin-top: 16px;
    padding: 13px 14px;
    border: 1px solid rgba(37,140,255,.42);
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(8,38,90,.68), rgba(4,17,45,.72));
}
.mf-side-info-title {
    color: #38bdf8 !important;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}
.mf-side-info-text {
    color: #a9c2df !important;
    font-size: 11px;
    margin-top: 4px;
}
.mf-side-privacy {
    display: flex;
    gap: 10px;
    align-items: center;
    margin-top: 10px;
    padding: 13px 14px;
    border: 1px solid rgba(96,165,250,.15);
    border-radius: 12px;
    background: rgba(3,12,30,.72);
}
.mf-privacy-icon { font-size: 22px; color: #38bdf8 !important; }
.mf-privacy-title { color: #e5eefb !important; font-size: 11px; font-weight: 700; }
.mf-privacy-text { color: #7186a1 !important; font-size: 10px; margin-top: 2px; }

/* ------------------------- HERO ------------------------- */
.mf-topbar {
    position: relative;
    overflow: hidden;
    min-height: 138px;
    padding: 22px 28px;
    margin-bottom: 22px;
    border: 1px solid rgba(37,99,235,.42);
    border-radius: 20px;
    background:
        radial-gradient(ellipse at 72% 50%, rgba(14,165,255,.13), transparent 35%),
        linear-gradient(110deg, rgba(3,10,29,.98), rgba(4,16,43,.90));
    box-shadow: 0 18px 60px rgba(0,0,0,.28), inset 0 1px rgba(255,255,255,.035);
}
.mf-topbar::after {
    content: "";
    position: absolute;
    right: -5%;
    top: 10%;
    width: 55%;
    height: 85%;
    background: repeating-radial-gradient(
        ellipse at center,
        transparent 0 11px,
        rgba(6,214,245,.07) 12px 13px,
        transparent 14px 24px
    );
    opacity: .65;
    transform: rotate(-7deg);
    pointer-events: none;
}
.mf-brand-wrap {
    display: flex;
    align-items: center;
    gap: 18px;
    position: relative;
    z-index: 2;
}
.mf-brand-icon {
    width: 84px;
    height: 84px;
    object-fit: contain;
    filter: drop-shadow(0 0 16px rgba(6,214,245,.17));
}
.mf-brand {
    font-size: 37px;
    font-weight: 800;
    letter-spacing: -1.7px;
    color: #ffffff;
}
.mf-brand span { color: #06d6f5; }
.mf-subtitle {
    color: #b7c8dd;
    font-size: 14px;
    margin-top: 6px;
}
.mf-badge {
    position: absolute;
    top: 23px;
    right: 24px;
    z-index: 3;
    padding: 9px 15px;
    border-radius: 999px;
    background: rgba(14,165,255,.07);
    color: #38bdf8;
    font-size: 12px;
    font-weight: 800;
    border: 1px solid rgba(14,165,255,.42);
}

/* ------------------------- TABS ------------------------- */
div[data-testid="stTabs"] > div:first-child {
    gap: 5px;
    background: transparent;
    padding: 0 0 8px;
    border-bottom: 1px solid rgba(148,163,184,.18);
}
div[data-testid="stTabs"] button {
    border-radius: 9px 9px 0 0 !important;
    padding: 11px 18px !important;
    font-weight: 600 !important;
    color: #8ea2bb !important;
    background: transparent !important;
}
div[data-testid="stTabs"] button:hover {
    color: #e0f2fe !important;
    background: rgba(37,99,235,.08) !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ffffff !important;
    background: rgba(11,45,115,.58) !important;
    box-shadow: inset 0 -2px #06d6f5, 0 0 20px rgba(37,99,235,.12);
}

/* ------------------------- CONTENT CARDS ------------------------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background:
        linear-gradient(145deg, rgba(4,13,32,.96), rgba(2,8,23,.96)) !important;
    border: 1px solid rgba(66,153,225,.22) !important;
    border-radius: 18px !important;
    box-shadow: 0 18px 45px rgba(0,0,0,.22) !important;
}

.mf-section {
    margin: 2px 0 9px;
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
.stCaption, .stMarkdown p, .stMarkdown li { color: #b8c7d9 !important; }

div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-testid="stDateInput"] input {
    background: #050f26 !important;
    color: #f8fafc !important;
    border: 1px solid rgba(96,165,250,.27) !important;
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
    border: 1px solid rgba(96,165,250,.27) !important;
    border-radius: 11px !important;
}
div[data-testid="stFileUploader"] section {
    background: #050f26 !important;
    border: 1px dashed rgba(96,165,250,.34) !important;
    border-radius: 14px !important;
}
div[data-testid="stFileUploader"] section * { color: #cbd5e1 !important; }

div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button {
    border-radius: 11px !important;
    border: 1px solid rgba(96,165,250,.25) !important;
    background: #07152f !important;
    color: #e0f2fe !important;
    font-weight: 700 !important;
    min-height: 42px;
}
div[data-testid="stButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:hover {
    border-color: #06b6d4 !important;
    color: #ffffff !important;
    box-shadow: 0 0 22px rgba(6,182,212,.14);
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg,#0b4cc9,#087eea) !important;
    color: #ffffff !important;
    border: 1px solid #1998ff !important;
    box-shadow: 0 8px 28px rgba(14,116,255,.25);
}
div[data-testid="stMetric"] {
    background: linear-gradient(145deg,#07152f,#030b1e) !important;
    border: 1px solid rgba(96,165,250,.20) !important;
    border-radius: 15px;
}
div[data-testid="stMetric"] label { color: #94a3b8 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f8fafc !important; }
div[data-testid="stExpander"] {
    border: 1px solid rgba(96,165,250,.20) !important;
    border-radius: 14px !important;
    background: rgba(4,13,32,.72) !important;
}
div[data-testid="stExpander"] summary { color: #e2e8f0 !important; }
.stAlert {
    background: #07152f !important;
    border: 1px solid rgba(96,165,250,.22) !important;
    color: #dbeafe !important;
}
.mf-footer {
    margin-top: 35px;
    padding: 18px;
    text-align: center;
    color: #64748b;
    font-size: 12px;
}
</style>

""", unsafe_allow_html=True)

# Embed the MeetFlow logo directly into the app so it also works on Streamlit Cloud.
MEETFLOW_ICON_B64 = "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"
MEETFLOW_ICON_DATA = "data:image/png;base64," + MEETFLOW_ICON_B64

# =========================== MEETFLOW BRAND ==============================
st.markdown(
    f"""
    <div class="mf-topbar">
        <div class="mf-brand-wrap">
            <img class="mf-brand-icon" src="{MEETFLOW_ICON_DATA}">
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

# Sidebar navigation is visual only; all existing controls and logic remain below.
st.sidebar.markdown(
    f"""
    <div class="mf-side-brand">
        <img src="{MEETFLOW_ICON_DATA}" class="mf-side-icon">
        <div class="mf-side-wordmark">Meet<span>Flow</span></div>
        <div class="mf-side-kicker">MEETING INTELLIGENCE</div>
    </div>

    <div class="mf-side-nav">
        <div class="mf-side-nav-item active">▦ &nbsp; AI Workspace</div>
        <div class="mf-side-nav-item">◷ &nbsp; History</div>
        <div class="mf-side-nav-item">⊞ &nbsp; Team Dashboard</div>
        <div class="mf-side-nav-item">⚙ &nbsp; Settings</div>
    </div>

    <div class="mf-side-divider"></div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="mf-side-section">WORKSPACE SETTINGS</div>', unsafe_allow_html=True)


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
    '<div class="mf-side-config-title">⚙️ Workspace Settings</div>',
    unsafe_allow_html=True,
)

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

st.sidebar.markdown(
    f"""
    <div class="mf-side-info">
        <div class="mf-side-info-title">✦ AI WORKSPACE</div>
        <div class="mf-side-info-text">
            {"Gemini connected and ready." if GOOGLE_API_KEY else "Add your Gemini API key to activate AI."}
        </div>
    </div>
    <div class="mf-side-privacy">
        <div class="mf-privacy-icon">♢</div>
        <div>
            <div class="mf-privacy-title">Your data stays private</div>
            <div class="mf-privacy-text">Meeting history is stored locally by the app.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)



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
    with st.container(border=True):
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
    with st.container(border=True):
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
    with st.container(border=True):
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
    with st.container(border=True):
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
    with st.container(border=True):
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
