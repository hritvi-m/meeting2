"""
AI Meeting Assistant
=====================
Upload meeting audio or notes -> get a summary, extracted & assigned action
items, a draft follow-up email, automatic Minutes of Meeting (PDF),
multilingual (English/Hindi/Hinglish) output, a chat interface to ask
questions about a meeting, and a team performance dashboard.

Install dependencies:
    pip install -r requirements.txt

Run:
    streamlit run app.py

You will need a Google Gemini API key (https://aistudio.google.com/apikey).

NOTE ON DEPLOYMENT:
This app must run on Python 3.11 or 3.12. Use a supported Python version such as 3.11 or 3.12 for the Streamlit/LangChain stack;
check your hosting provider if you encounter dependency compatibility errors.
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
import uuid
from datetime import datetime, date

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from fpdf import FPDF

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# =========================== STEP 2: APP CONFIG =============================

st.set_page_config(page_title="AI Meeting Assistant", layout="wide")
st.title("🗓️ AI Meeting Assistant")
st.caption(
    "Upload meeting audio or notes → summary, action items, MoM PDF, "
    "multilingual output, meeting chat, and a team performance dashboard."
)

HISTORY_FILE = "meeting_history.json"
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

st.sidebar.title("⚙️ Configuration")

GOOGLE_API_KEY = st.sidebar.text_input("GOOGLE_API_KEY", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY

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
    if not GOOGLE_API_KEY:
        raise RuntimeError("Google Gemini API key is missing.")
    return ChatGoogleGenerativeAI(
        model=model_choice,
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY,
    )


def transcribe_audio(file_path: str) -> str:
    """Transcribe meeting audio using the current Google GenAI SDK.

    The legacy google-generativeai SDK is intentionally not used here.
    Audio is uploaded with google-genai's Files API and sent to Gemini
    for transcription. The original spoken language is preserved.
    """
    if not GOOGLE_API_KEY:
        raise RuntimeError("Google Gemini API key is missing.")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    try:
        uploaded_file = client.files.upload(file=file_path)

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[
                "Transcribe this meeting recording as accurately as possible. "
                "The speakers may talk in English, Hindi, or Hinglish (code-switched "
                "Hindi-English). Preserve the language and script actually spoken; "
                "do not translate. Label speakers as Speaker 1, Speaker 2, etc. if "
                "they are distinguishable. Return only the transcript text.",
                uploaded_file,
            ],
        )
    except Exception as exc:
        raise RuntimeError(
            "Audio transcription failed. Please check that your Gemini API key is valid "
            "and that the selected project has Gemini API access. Original error: "
            f"{exc}"
        ) from exc

    transcript = getattr(response, "text", None)
    if not transcript:
        raise RuntimeError("Gemini returned an empty transcript for the uploaded audio.")

    return transcript.strip()


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


def generate_mom_pdf(
    meeting_title: str,
    meeting_date: str,
    summary: str,
    action_items: list
) -> bytes:
    """Generate a professional Minutes of Meeting PDF.

    Supports:
    - English
    - Hindi / Devanagari
    - Hinglish
    - Automatic line wrapping
    - Multiple pages
    - Proper left/right margins
    """

    pdf = FPDF("P", "mm", "A4")

    # Page margins
    pdf.set_margins(18, 18, 18)

    # Automatic page break
    pdf.set_auto_page_break(auto=True, margin=18)

    # ---------------------------------------------------------
    # FONT
    # ---------------------------------------------------------
    if UNICODE_FONT_AVAILABLE:
        pdf.add_font(
            "NotoSans",
            "",
            FONT_REGULAR
        )
        pdf.add_font(
            "NotoSans",
            "B",
            FONT_BOLD
        )
        font_name = "NotoSans"
    else:
        font_name = "Helvetica"

    pdf.add_page()

    page_width = pdf.epw

    # ---------------------------------------------------------
    # TITLE
    # ---------------------------------------------------------
    pdf.set_font(font_name, "B", 18)

    # IMPORTANT:
    # new_x="LMARGIN" makes sure the next line starts
    # from the left margin instead of the right side.
    pdf.multi_cell(
        page_width,
        10,
        "MINUTES OF MEETING",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.ln(4)

    # ---------------------------------------------------------
    # MEETING INFORMATION
    # ---------------------------------------------------------
    pdf.set_font(font_name, "B", 11)

    pdf.multi_cell(
        page_width,
        7,
        _sanitize_for_pdf("Meeting Title"),
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.set_font(font_name, "", 11)

    pdf.multi_cell(
        page_width,
        7,
        _sanitize_for_pdf(meeting_title),
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.multi_cell(
        page_width,
        7,
        _sanitize_for_pdf(f"Date: {meeting_date}"),
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.ln(5)

    # ---------------------------------------------------------
    # SUMMARY HEADING
    # ---------------------------------------------------------
    pdf.set_font(font_name, "B", 14)

    pdf.multi_cell(
        page_width,
        8,
        _sanitize_for_pdf("Summary"),
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.ln(1)

    # ---------------------------------------------------------
    # SUMMARY CONTENT
    # ---------------------------------------------------------
    pdf.set_font(font_name, "", 10.5)

    clean_summary = _sanitize_for_pdf(summary)

    # Split paragraphs so spacing is better
    summary_paragraphs = clean_summary.split("\n")

    for paragraph in summary_paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            pdf.ln(2)
            continue

        pdf.multi_cell(
            page_width,
            6.5,
            paragraph,
            align="L",
            new_x="LMARGIN",
            new_y="NEXT"
        )

    pdf.ln(5)

    # ---------------------------------------------------------
    # ACTION ITEMS HEADING
    # ---------------------------------------------------------
    pdf.set_font(font_name, "B", 14)

    pdf.multi_cell(
        page_width,
        8,
        _sanitize_for_pdf("Action Items"),
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.ln(2)

    # ---------------------------------------------------------
    # ACTION ITEMS
    # ---------------------------------------------------------
    if action_items:

        for i, item in enumerate(action_items, 1):

            task = item.get("task", "").strip()
            owner = item.get("owner", "Unassigned")
            deadline = item.get("deadline", "TBD")
            status = item.get("status", "Pending")

            # Task
            pdf.set_font(font_name, "B", 10.5)

            task_text = _sanitize_for_pdf(
                f"{i}. {task}"
            )

            pdf.multi_cell(
                page_width,
                6.5,
                task_text,
                align="L",
                new_x="LMARGIN",
                new_y="NEXT"
            )

            # Owner / deadline / status
            pdf.set_font(font_name, "", 10)

            details = _sanitize_for_pdf(
                f"Owner: {owner}    |    "
                f"Due: {deadline}    |    "
                f"Status: {status}"
            )

            pdf.multi_cell(
                page_width,
                6,
                details,
                align="L",
                new_x="LMARGIN",
                new_y="NEXT"
            )

            pdf.ln(3)

    else:

        pdf.set_font(font_name, "", 10.5)

        pdf.multi_cell(
            page_width,
            6.5,
            "No action items were identified.",
            new_x="LMARGIN",
            new_y="NEXT"
        )

    # ---------------------------------------------------------
    # FOOTER
    # ---------------------------------------------------------
    pdf.set_y(-15)

    pdf.set_font(font_name, "", 8)

    pdf.cell(
        0,
        5,
        "Generated by AI Meeting Assistant",
        align="C"
    )

    # ---------------------------------------------------------
    # RETURN PDF
    # ---------------------------------------------------------
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

tab_new, tab_chat, tab_history, tab_dashboard = st.tabs(
    ["📝 New Meeting", "💬 Ask Your Meeting", "📚 Meeting History", "📈 Team Dashboard"]
)

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
        # Clear any previous successful result before starting a new attempt.
        # This prevents an old meeting from being displayed as "processed"
        # when the new request fails.
        st.session_state.pop("current_result", None)

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
    st.subheader("💬 Ask Your Meeting")
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
    st.subheader("📈 Team Performance Dashboard")
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
