import streamlit as st
import datetime
import json
import csv
import numpy as np
import faiss
import langid
from openai import OpenAI
from pathlib import Path

# --- CONFIG ---
LOGO_PATH = "assets/logo-haw-hamburg.gif"
LOGO_STATIC = "assets/haw-logo.png"
HAW_BLUE = "#004B87"
HAW_DARK = "#003567"
HAW_LIGHT = "#E6F0FA"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

def get_openai_client():
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None
    if not api_key:
        st.error("OpenAI API key not configured.")
        st.markdown("Create `.streamlit/secrets.toml` in the project root with:")
        st.code('OPENAI_API_KEY = "sk-..."', language="toml")
        st.markdown("Then restart the app.")
        st.stop()
    return OpenAI(api_key=api_key)

client = get_openai_client()

st.set_page_config(
    page_title="AskHAW",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD INDEXES ---
@st.cache_resource
def load_indexes():
    def load_one(lang):
        index = faiss.read_index(f"data/haw_kb_{lang}.index")
        with open(f"data/haw_kb_meta_{lang}.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        return index, meta
    en_index, en_meta = load_one("en")
    de_index, de_meta = load_one("de")
    return {"en": (en_index, en_meta), "de": (de_index, de_meta)}

indexes = load_indexes()

# --- CUSTOM CSS ---
st.markdown(f"""
<style>
/* ---- Global ---- */
#MainMenu, footer {{ visibility: hidden; }}
.stApp {{ background-color: #F0F4F9; }}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {HAW_DARK} 0%, {HAW_BLUE} 100%);
    padding-top: 0;
}}
[data-testid="stSidebar"] section {{
    padding-top: 1rem;
}}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {{
    color: white !important;
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: white !important;
}}
[data-testid="stSidebar"] .stSelectbox > div > div {{
    background-color: rgba(255,255,255,0.15) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    border-radius: 10px !important;
    color: white !important;
}}
[data-testid="stSidebar"] .stSelectbox svg {{
    fill: white !important;
}}
[data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.2);
}}

/* ---- Clear Chat button in sidebar ---- */
[data-testid="stSidebar"] .stButton button {{
    background-color: rgba(255,255,255,0.15) !important;
    color: white !important;
    border: 1.5px solid rgba(255,255,255,0.5) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}}
[data-testid="stSidebar"] .stButton button:hover {{
    background-color: rgba(255,255,255,0.25) !important;
}}

/* ---- Welcome banner title fix ---- */
.welcome-banner h2 {{
    color: white !important;
}}

/* ---- Header card ---- */
.haw-header {{
    background: white;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 12px rgba(0,75,135,0.10);
    border-left: 5px solid {HAW_BLUE};
    display: flex;
    align-items: center;
    gap: 1rem;
}}
.haw-title {{
    font-size: 1.9rem;
    font-weight: 800;
    color: {HAW_BLUE};
    margin: 0;
    letter-spacing: -0.5px;
}}
.haw-subtitle {{
    font-size: 0.88rem;
    color: #64748B;
    margin: 0.2rem 0 0 0;
}}

/* ---- Welcome banner ---- */
.welcome-banner {{
    background: linear-gradient(135deg, {HAW_BLUE} 0%, #0069C0 100%);
    color: white;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.2rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,75,135,0.20);
}}
.welcome-banner h2 {{
    margin: 0 0 0.5rem 0;
    font-size: 1.3rem;
    font-weight: 700;
}}
.welcome-banner p {{
    margin: 0;
    font-size: 0.9rem;
    opacity: 0.9;
    line-height: 1.5;
}}

/* ---- Suggestion chips (st.button styled) ---- */
div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"] .stButton button {{
    background: white !important;
    border: 1.5px solid {HAW_BLUE} !important;
    color: {HAW_BLUE} !important;
    border-radius: 20px !important;
    padding: 0.35rem 0.6rem !important;
    font-size: 0.80rem !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    white-space: normal !important;
    line-height: 1.3 !important;
    height: auto !important;
    min-height: 2.4rem !important;
}}
div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"] .stButton button:hover {{
    background: {HAW_LIGHT} !important;
    border-color: {HAW_DARK} !important;
    color: {HAW_DARK} !important;
}}

/* ---- Chat messages ---- */
[data-testid="stChatMessage"] {{
    background: white;
    border-radius: 16px;
    border: 1px solid #E2ECF6;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    margin-bottom: 0.6rem;
    padding: 0.2rem 0.5rem;
}}

/* ---- Chat input ---- */
[data-testid="stChatInputTextArea"] {{
    border-radius: 24px !important;
    font-size: 0.95rem !important;
}}
[data-testid="stChatInput"] {{
    border: 2px solid {HAW_BLUE} !important;
    border-radius: 28px !important;
    background: white !important;
    box-shadow: 0 2px 12px rgba(0,75,135,0.10) !important;
    padding: 0.2rem 0.5rem !important;
}}
[data-testid="stChatInput"] button {{
    background-color: {HAW_BLUE} !important;
    border-radius: 50% !important;
    color: white !important;
}}

/* ---- Feedback buttons ---- */
div[data-testid="stHorizontalBlock"]:has(button[title="Helpful"]) button,
div[data-testid="stHorizontalBlock"]:has(button[title="Not helpful"]) button {{
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}}

/* ---- Footer ---- */
.haw-footer {{
    text-align: center;
    font-size: 0.78rem;
    color: #94A3B8;
    margin-top: 1rem;
    padding-top: 0.8rem;
    border-top: 1px solid #E2ECF6;
}}
</style>
""", unsafe_allow_html=True)

# --- PERSONA ---
PERSONA = (
    "You are AskHAW, a warm and friendly assistant for Hamburg University of Applied Sciences (HAW Hamburg). "
    "Speak in a helpful, approachable tone. Use simple language and feel free to add a friendly remark where fitting."
)

# --- GDPR CONSENT ---
if "gdpr_accepted" not in st.session_state:
    st.session_state.gdpr_accepted = False

if not st.session_state.gdpr_accepted:
    st.markdown("<div style='max-width:680px; margin:4rem auto;'>", unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 6, 1])
    with col_c:
        st.image(LOGO_STATIC, width=120)
        st.markdown(f"""
        <div style="
            background:white;
            border-radius:20px;
            border-left: 5px solid {HAW_BLUE};
            box-shadow: 0 4px 24px rgba(0,75,135,0.12);
            padding: 2rem 2.2rem;
            margin-top: 1rem;
        ">
            <h2 style="color:{HAW_BLUE}; margin-top:0;">Privacy Notice</h2>
            <p style="color:#374151; line-height:1.7;">
                Welcome to <strong>AskHAW</strong>, the AI assistant for HAW Hamburg.
                Before you start, please read how your data is handled:
            </p>
            <ul style="color:#374151; line-height:2; padding-left:1.2rem;">
                <li>Your questions are sent to <strong>OpenAI's API</strong> to generate answers.</li>
                <li><strong>No personal data</strong> (name, student ID, email) is collected or stored by AskHAW.</li>
                <li>Your conversation exists only for the <strong>current session</strong> and is cleared when you close or refresh the page.</li>
                <li>AskHAW is a <strong>prototype</strong> and not an official HAW Hamburg service.</li>
                <li>You can <strong>clear your chat</strong> at any time using the button in the sidebar.</li>
            </ul>
            <p style="color:#6B7280; font-size:0.85rem; margin-bottom:0;">
                By clicking <em>Accept & Continue</em> you confirm that you have read and understood this notice.
                For HAW Hamburg's full privacy policy, visit
                <a href="https://www.haw-hamburg.de/en/data-privacy-policy/" target="_blank" style="color:{HAW_BLUE};">haw-hamburg.de/en/data-privacy-policy</a>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.2rem;'>", unsafe_allow_html=True)
        if st.button("✓  Accept & Continue", use_container_width=True, type="primary"):
            st.session_state.gdpr_accepted = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f'<p style="text-align:center; color:#9CA3AF; font-size:0.78rem; margin-top:1rem;">'
            f'AskHAW Prototype &nbsp;|&nbsp; HAW Hamburg {datetime.date.today().year} &nbsp;|&nbsp; Not an official HAW service</p>',
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.image(LOGO_STATIC, width=140)
    st.markdown("## AskHAW")
    st.markdown("Your HAW Hamburg assistant")
    st.markdown("---")

    st.markdown("---")
    st.markdown("#### Quick Tips")
    st.markdown("""
- Ask about **courses & programs**
- Check **exam deadlines**
- Find **scholarship** info
- Get **contact details**
- Learn about **campus life**
    """)

    st.markdown("---")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("Privacy Notice", use_container_width=True):
        st.session_state.gdpr_accepted = False
        st.session_state.chat_history = []
        st.rerun()

    st.markdown(
        f'<div style="margin-top:2rem; font-size:0.78rem; opacity:0.6; text-align:center;">'
        f'AskHAW Prototype<br>HAW Hamburg {datetime.date.today().year}</div>',
        unsafe_allow_html=True
    )

# --- HEADER ---
with st.container():
    st.markdown(
        '<div class="haw-title">AskHAW</div>'
        '<div class="haw-subtitle">Your intelligent assistant for HAW Hamburg — courses, exams, housing, scholarships and more.</div>',
        unsafe_allow_html=True
    )

st.markdown("<div style='margin-bottom:1rem'></div>", unsafe_allow_html=True)

# --- CHAT STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "feedback" not in st.session_state:
    st.session_state.feedback = {}  # {msg_index: "positive" or "negative"}

# --- WELCOME BANNER (shown only on empty chat, and not while processing a chip click) ---
if not st.session_state.chat_history and not st.session_state.pending_prompt:
    st.markdown("""
    <div class="welcome-banner">
        <h2>Welcome to AskHAW!</h2>
        <p>Ask me anything about studying at HAW Hamburg — in English or German.<br>
        I can help with programs, deadlines, scholarships, campus life, and more.</p>
    </div>
    """, unsafe_allow_html=True)

    CHIPS = [
        "What programs does HAW offer?",
        "How do I apply for a Bachelor?",
        "Are there scholarships?",
        "Where is the library?",
        "How do I contact student counselling?",
    ]
    chip_cols = st.columns(len(CHIPS))
    for col, chip in zip(chip_cols, CHIPS):
        with col:
            if st.button(chip, key=f"chip_{chip}", use_container_width=True):
                st.session_state.pending_prompt = chip
                st.rerun()

# --- FEEDBACK HELPER ---
def save_feedback(question, answer, rating):
    feedback_file = Path("data/feedback.csv")
    is_new = not feedback_file.exists()
    with open(feedback_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "rating", "question", "answer"])
        writer.writerow([
            datetime.datetime.now().isoformat(),
            rating,
            question[:300],
            answer[:600],
        ])

# --- DISPLAY CHAT HISTORY ---
for i, (role, message) in enumerate(st.session_state.chat_history):
    avatar = LOGO_PATH if role == "assistant" else None
    with st.chat_message(role, avatar=avatar):
        st.markdown(message)

    # Feedback buttons after each assistant message
    if role == "assistant":
        if i in st.session_state.feedback:
            emoji = "👍" if st.session_state.feedback[i] == "positive" else "👎"
            st.markdown(
                f'<div style="font-size:0.78rem; color:#9CA3AF; margin:-0.4rem 0 0.6rem 0.5rem;">'
                f'{emoji} Thanks for your feedback!</div>',
                unsafe_allow_html=True
            )
        else:
            fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 18])
            with fb_col1:
                if st.button("👍", key=f"up_{i}", help="Helpful"):
                    question = st.session_state.chat_history[i - 1][1] if i > 0 else ""
                    save_feedback(question, message, "positive")
                    st.session_state.feedback[i] = "positive"
                    st.rerun()
            with fb_col2:
                if st.button("👎", key=f"down_{i}", help="Not helpful"):
                    question = st.session_state.chat_history[i - 1][1] if i > 0 else ""
                    save_feedback(question, message, "negative")
                    st.session_state.feedback[i] = "negative"
                    st.rerun()

# --- FUNCTIONS ---
def search_kb(query, lang):
    index, meta = indexes[lang]
    records = meta["records"]
    q_embed = client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
    q_vec = np.array(q_embed, dtype="float32").reshape(1, -1)
    D, I = index.search(q_vec, k=5)
    hits = [records[i] for i in I[0] if 0 <= i < len(records)]
    matches = [hit["text"] for hit in hits]
    return matches, D[0][0], hits

def build_retrieval_query(query, chat_history):
    return query

def build_messages(query, lang, chat_history):
    retrieval_query = build_retrieval_query(query, chat_history)
    matches, score, hits = search_kb(retrieval_query, lang)
    if not matches or score > 3.5:
        return None, False, []

    context = "\n\n".join(matches)
    persona = PERSONA

    system_prompt = f"""{persona}

STRICT RULES — you MUST follow these without exception:
1. Answer using ONLY the information explicitly stated in the context below.
2. NEVER invent, guess, or assume any facts — including program names, course titles, department names, email addresses, deadlines, or any other details not present in the context.
3. If the context does not contain enough information to fully answer the question, say clearly: "I only have partial information on this. Based on what I know: [answer what you can]. For complete details, please visit haw-hamburg.de or contact the Student Counselling Office at studienberatung@haw-hamburg.de."
4. If the context contains NO relevant information, say: "I don't have enough information to answer that accurately. Please check haw-hamburg.de or contact studienberatung@haw-hamburg.de for reliable details."
5. Do NOT present partial or uncertain information as if it is complete or confirmed.

Context:
{context}"""

    messages = [{"role": "system", "content": system_prompt}]
    for role, content in chat_history[:-1]:
        api_role = "user" if role == "user" else "assistant"
        messages.append({"role": api_role, "content": content})
    messages.append({"role": "user", "content": query})
    return messages, True, hits

# --- FALLBACK ---
def get_fallback(lang):
    if lang == "de":
        return (
            "Das konnte ich leider nicht in meiner Wissensdatenbank finden. "
            "Fur weitere Hilfe kannst du:\n"
            "- Die HAW Hamburg Website besuchen: [haw-hamburg.de](https://www.haw-hamburg.de)\n"
            "- Die Studienberatung kontaktieren: studienberatung@haw-hamburg.de / +49 151 7281 8022\n"
            "- Das Studierendensekretariat erreichen: studierendensekretariat@haw-hamburg.de / +49 40 42875 9898"
        )
    return (
        "I couldn't find that in my knowledge base. For more help:\n"
        "- Browse the HAW Hamburg website: [haw-hamburg.de](https://www.haw-hamburg.de)\n"
        "- Contact Student Counselling: studienberatung@haw-hamburg.de / +49 151 7281 8022\n"
        "- Reach the Admissions Office: studierendensekretariat@haw-hamburg.de / +49 40 42875 9898"
    )

# --- SOURCES RENDERING ---
def render_sources(hits):
    if not hits:
        return
    seen = set()
    unique = []
    for h in hits:
        url = h.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(h)
    if not unique:
        return
    links = []
    for h in unique:
        title = h.get("title") or h.get("url")
        url = h.get("url")
        links.append(f"- [{title}]({url})")
    st.markdown(
        '<div style="font-size:0.8rem; color:#64748B; margin-top:0.4rem;">Sources:</div>',
        unsafe_allow_html=True
    )
    st.markdown("\n".join(links))
    with st.expander("View retrieved passages"):
        for h in unique:
            st.markdown(f"**{h.get('title') or h.get('url')}**")
            st.caption(h.get("url", ""))
            text = h.get("text", "")
            st.write(text[:600] + ("…" if len(text) > 600 else ""))

# --- CHAT INPUT ---
def handle_prompt(prompt):
    """Process a user prompt (from chat input or chip click)."""
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append(("user", prompt))

    lang, _ = langid.classify(prompt)
    lang = "de" if lang == "de" else "en"

    messages, valid, hits = build_messages(prompt, lang, st.session_state.chat_history)

    if not valid:
        response = get_fallback(lang)
        with st.chat_message("assistant", avatar=LOGO_PATH):
            st.markdown(response)
        st.session_state.chat_history.append(("assistant", response))
    else:
        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            stream=True
        )
        with st.chat_message("assistant", avatar=LOGO_PATH):
            response = st.write_stream(stream)
            if "I don't have enough information" not in response:
                render_sources(hits)
        st.session_state.chat_history.append(("assistant", response))

# Handle chip-click pending prompt
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    handle_prompt(prompt)

if prompt := st.chat_input("Ask me anything about HAW Hamburg..."):
    handle_prompt(prompt)

# --- FOOTER ---
st.markdown(
    f'<div class="haw-footer">AskHAW Prototype &nbsp;|&nbsp; HAW Hamburg {datetime.date.today().year} &nbsp;|&nbsp; '
    f'Not an official HAW service</div>',
    unsafe_allow_html=True
)
