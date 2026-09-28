import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

load_dotenv()

st.set_page_config(page_title="Mood Chat", page_icon="💬", layout="centered")

# ---------- Moods ----------
MOODS = {
    "Funny": {
        "emoji": "😄",
        "accent": "#F5A623",
        "prompt": "You are a funny AI agent",
        "tagline": "Expect jokes, puns and playful replies.",
    },
    "Sad": {
        "emoji": "😢",
        "accent": "#4F86C6",
        "prompt": "You are a sad, melancholic AI agent. Respond in a gloomy, sorrowful tone.",
        "tagline": "Gloomy, sighing and a little heartbroken.",
    },
    "Angry": {
        "emoji": "😠",
        "accent": "#E5484D",
        "prompt": "You are an angry, irritated AI agent. Respond in a grumpy, fed-up tone.",
        "tagline": "Grumpy, blunt and easily annoyed.",
    },
}


# ---------- Model ----------
@st.cache_resource
def get_model():
    return ChatGroq(model="openai/gpt-oss-20b", temperature=1)


# ---------- State ----------
if "history" not in st.session_state:
    st.session_state.history = []  # only human/AI turns


def reset_chat():
    """Runs the moment the mood is changed: start a fresh conversation."""
    st.session_state.history = []


# ---------- Mood picker ----------
mood_name = st.radio(
    "Choose the AI's mood",
    list(MOODS.keys()),
    format_func=lambda m: f"{MOODS[m]['emoji']}  {m}",
    horizontal=True,
    key="mood_choice",
    on_change=reset_chat,
)
mood = MOODS[mood_name]

# ---------- Styling (accent follows the mood) ----------
st.markdown(
    f"""
    <style>
        .block-container {{ max-width: 760px; padding-top: 2.5rem; }}
        .mood-title {{
            font-size: 2.1rem; font-weight: 700; margin: 0 0 .15rem 0;
            border-left: 6px solid {mood['accent']}; padding-left: .75rem;
        }}
        .mood-sub {{ color: #8a8f98; margin: 0 0 1.25rem .95rem; }}
        div[data-testid="stChatInput"] textarea:focus {{
            border-color: {mood['accent']} !important;
            box-shadow: 0 0 0 1px {mood['accent']} !important;
        }}
        div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {{
            background-color: {mood['accent']} !important;
            border-color: {mood['accent']} !important;
        }}
    </style>
    <p class="mood-title">Mood Chat</p>
    <p class="mood-sub">{mood['emoji']} {mood['tagline']}</p>
    """,
    unsafe_allow_html=True,
)

# ---------- Show conversation ----------
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    else:
        with st.chat_message("assistant", avatar=mood["emoji"]):
            st.write(msg.content)

# ---------- Chat input ----------
prompt = st.chat_input(f"Message the {mood_name.lower()} AI...")

if prompt:
    st.session_state.history.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    messages = [SystemMessage(content=mood["prompt"])] + st.session_state.history

    with st.chat_message("assistant", avatar=mood["emoji"]):
        try:
            with st.spinner("Thinking..."):
                response = get_model().invoke(messages)
            st.write(response.content)
            st.session_state.history.append(AIMessage(content=response.content))
        except Exception as e:
            st.error(f"Could not get a reply: {e}")
            st.session_state.history.pop()  # drop the unanswered message