import streamlit as st
import random
import pandas as pd
import os
import spacy
from datetime import datetime
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load intents
df_intents = pd.read_csv("intents.csv")

# Initialize chat log file if not present
if not os.path.exists("chat_logs.csv"):
    pd.DataFrame(columns=["Timestamp", "User", "Bot", "Intent", "Sentiment"]).to_csv("chat_logs.csv", index=False)

# --- Functions ---

def get_intent(user_input):
    """Finds intent based on pattern matching."""
    user_input = user_input.lower()
    for _, row in df_intents.iterrows():
        patterns = str(row["patterns"]).split("|")
        for p in patterns:
            if p.strip() in user_input:
                return row["intent"], row["responses"]
    return "unknown", "I'm not sure I understand. Could you rephrase?"

def analyze_sentiment(text):
    """Uses TextBlob to detect sentiment polarity."""
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0.2:
        return "Positive"
    elif blob.sentiment.polarity < -0.2:
        return "Negative"
    else:
        return "Neutral"

def get_response(intent, responses):
    """Returns a formatted chatbot response."""
    possible_responses = str(responses).split("|")
    response = random.choice(possible_responses)
    response = response.replace("{time_now}", datetime.now().strftime("%I:%M %p"))
    response = response.replace("{date_now}", datetime.now().strftime("%B %d, %Y"))
    return response

def save_chat_log(user, bot, intent, sentiment):
    """Appends each conversation to CSV log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_log = pd.DataFrame([[timestamp, user, bot, intent, sentiment]],
                           columns=["Timestamp", "User", "Bot", "Intent", "Sentiment"])
    new_log.to_csv("chat_logs.csv", mode="a", index=False, header=False)

# --- Streamlit UI ---

st.set_page_config(page_title="Dynamic AI Chatbot", layout="wide")
st.title("🤖 DYNAMIC AI CHATBOT  (A Simple but Dynamic Bot)")

menu = ["💬 Chat", "📊 Chat Analytics Dashboard"]
choice = st.sidebar.selectbox("Select Mode", menu)

# --- Chat Mode ---
if choice == "💬 Chat":
    st.markdown("Start chatting with your AI assistant below 👇")

    if "context" not in st.session_state:
        st.session_state.context = []

    user_input = st.text_input("You:", "")

    if st.button("Send") and user_input.strip():
        intent, responses = get_intent(user_input)
        sentiment = analyze_sentiment(user_input)
        bot_response = get_response(intent, responses)

        # Save to log
        save_chat_log(user_input, bot_response, intent, sentiment)

        # Display conversation
        st.markdown(f"**💬 You:** {user_input}")
        st.markdown(f"**🤖 Bot:** {bot_response}")
        st.markdown(f"🧾 Intent: `{intent}` | 🧠 Sentiment: `{sentiment}`")

# --- Analytics Mode ---
elif choice == "📊 Chat Analytics Dashboard":
    st.subheader("Chat Analytics Dashboard")

    # Safe load
    if os.path.exists("chat_logs.csv") and os.path.getsize("chat_logs.csv") > 0:
        logs = pd.read_csv("chat_logs.csv")
    else:
        logs = pd.DataFrame(columns=["Timestamp", "User", "Bot", "Intent", "Sentiment"])

    if logs.empty:
        st.warning("No chat data available yet. Chat first to generate analytics.")
    else:
        st.dataframe(logs.tail(10))

        st.markdown("### 🔹 Sentiment Distribution")
        sentiment_counts = logs["Sentiment"].value_counts()
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', ax=ax)
        st.pyplot(fig)

        st.markdown("### 🔹 Most Common Intents")
        intent_counts = logs["Intent"].value_counts()
        fig2, ax2 = plt.subplots()
        intent_counts.plot(kind='barh', ax=ax2)
        st.pyplot(fig2)

        st.success("✅ Analytics Generated Successfully!")
