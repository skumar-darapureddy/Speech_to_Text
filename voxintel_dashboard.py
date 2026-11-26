import streamlit as st
import pandas as pd
import io
import ast

from voxintel.analytics.conversation_analytics import ConversationAnalytics
from voxintel.knowledge_graph.knowledge_graph import KnowledgeGraph
from voxintel.intelligence.speaker_detection import SpeakerDetector
from voxintel.language.language_detection import LanguageDetector
from voxintel.intelligence.event_extraction import EventExtractor
from voxintel.memory.conversation_memory import ConversationMemory
from voxintel.database.conversation_database import ConversationDatabase

from data_science.eda import EDAAnalyzer
from data_science.statistics import StatisticsAnalyzer
from data_science.visualization import VisualizationEngine
from data_science.ml_models import MLAnalyzer
from nlp.text_analysis import TextAnalyzer
from nlp.sentiment import SentimentAnalyzer
from nlp.advanced_nlp import AdvancedNLP
from business.business_analytics import BusinessAnalytics
from business.customer_segmentation import CustomerSegmentation
from business.fraud_detection import FraudDetector
from business.sales_forecasting import SalesForecaster


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VOXINTEL",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)



# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 46px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 4px;
}

.subtitle {
    font-size: 18px;
    opacity: 0.72;
    margin-bottom: 28px;
}

.section-title {
    font-size: 28px;
    font-weight: 750;
    margin-top: 10px;
    margin-bottom: 18px;
}

div[data-testid="stMetric"] {
    border-radius: 16px;
    padding: 18px 20px;
    border: 1px solid rgba(128,128,128,0.18);
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
}

div[data-testid="stMetricValue"] {
    font-size: 30px;
    font-weight: 750;
}

div[data-testid="stMetricLabel"] {
    font-size: 14px;
    font-weight: 600;
    opacity: 0.78;
}

.stButton > button {
    border-radius: 10px;
    font-weight: 650;
    padding: 8px 18px;
}

[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.18);
}

.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
}

div[data-testid="stExpander"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)
# ============================================================
# DATABASE
# ============================================================

def load_conversations(database):
    columns = [
        "id", "timestamp", "transcript", "language", "speaker",
        "events", "sentiment", "polarity", "subjectivity",
        "confidence", "reviewed_sentiment", "review_notes",
        "reviewed_at"
    ]

    rows = database.get_all_conversations()
    return pd.DataFrame(rows, columns=columns)


def parse_events(value):
    if not value:
        return []
    try:
        events = ast.literal_eval(str(value))
        return events if isinstance(events, list) else []
    except Exception:
        return []


def event_counts(df):
    counter = {}
    if df.empty:
        return counter

    for value in df["events"]:
        for event in parse_events(value):
            if isinstance(event, dict):
                event_type = event.get("type", "Unknown")
            else:
                event_type = str(event)
            counter[event_type] = counter.get(event_type, 0) + 1

    return counter


def display_event_cards(events):
    """Display polished event cards while filtering duplicate details."""

    if not events:
        st.info("No events detected.")
        return

    event_styles = {
        "Meeting": ("MEETING", "📅", "#2563EB"),
        "Reminder": ("REMINDER", "⏰", "#D97706"),
        "Call": ("CALL", "📞", "#059669"),
        "Task": ("TASK", "✅", "#7C3AED"),
        "Question": ("QUESTION", "❓", "#0891B2"),
        "Time": ("TIME", "🕒", "#64748B"),
        "Date": ("DATE", "📆", "#64748B"),
        "Location": ("LOCATION", "📍", "#64748B"),
        "Phone": ("PHONE", "📱", "#64748B"),
        "Amount": ("AMOUNT", "💰", "#64748B"),
    }

    field_labels = [
        ("date", "Date"),
        ("time", "Time"),
        ("location", "Location"),
        ("action", "Action"),
        ("person", "Person"),
        ("value", "Value"),
    ]

    meeting_details = set()

    for event in events:
        if not isinstance(event, dict):
            continue

        if str(event.get("type", "")).lower() == "meeting":
            for key in ("date", "time", "location"):
                value = event.get(key)
                if value not in (None, ""):
                    normalized = " ".join(str(value).lower().split())
                    meeting_details.add((key, normalized))

    for event in events:
        if not isinstance(event, dict):
            continue

        event_type = str(event.get("type", "Event"))

        if event_type in ("Date", "Time", "Location"):
            value = event.get("value")
            key = event_type.lower()

            if value not in (None, ""):
                normalized = " ".join(str(value).lower().split())
                if (key, normalized) in meeting_details:
                    continue

        title, icon, accent = event_styles.get(
            event_type,
            (event_type.upper(), "🔹", "#64748B")
        )

        details = [
            (label, event[key])
            for key, label in field_labels
            if event.get(key) not in (None, "")
        ]

        with st.container(border=True):
            st.markdown(
                f"""
                <div style="
                    border-left: 5px solid {accent};
                    padding: 3px 0 3px 12px;
                    margin-bottom: 12px;
                ">
                    <div style="
                        color: {accent};
                        font-size: 13px;
                        font-weight: 800;
                        letter-spacing: 1.2px;
                    ">{icon} {title}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if details:
                columns = st.columns(min(len(details), 3))

                for index, (label, value) in enumerate(details):
                    with columns[index % len(columns)]:
                        st.caption(label.upper())
                        st.markdown(f"**{value}**")

            elif event.get("text"):
                st.write(str(event["text"]))


# ============================================================
# INITIALIZE COMPONENTS
# ============================================================

database = ConversationDatabase()

try:
    df = load_conversations(database)
except Exception as e:
    st.error(f"Supabase database connection failed: {e}")
    st.stop()

analytics = ConversationAnalytics()
graph = KnowledgeGraph()
speaker_detector = SpeakerDetector()
language_detector = LanguageDetector()
event_extractor = EventExtractor()
memory = ConversationMemory()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎙️ VOXINTEL</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Multilingual Conversation & Customer Intelligence Platform'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎙️ VOXINTEL")
st.sidebar.markdown("### AI + Data Science Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "\U0001F4CA Dashboard",
        "\U0001F399\ufe0f Speech-to-Text",
        "\U0001F9E0 Conversation Intelligence",
        "\U0001F60A Sentiment & NLP",
        "\U0001F465 Customer Segmentation",
        "\U0001F6A8 Fraud Detection",
        "\U0001F4C8 Sales Forecasting",
        "\U0001F4BC Business Analytics",
        "\U0001F4CB EDA & Statistics",
        "\U0001F4CA Machine Learning",
        "\U0001F578 Knowledge Graph",
        "\U0001F5C4 Conversation Database",
        "\U0001F4CA Analytics"
]
)

st.sidebar.markdown("---")
st.sidebar.success("\U0001F7E2 System Online")
st.sidebar.metric("Conversations", len(df))

total_words_sidebar = (
    sum(len(str(x).split()) for x in df["transcript"])
    if not df.empty else 0
)
st.sidebar.metric("Total Words", total_words_sidebar)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Speech \u2192 NLP \u2192 Intelligence \u2192 Data Science \u2192 Business Insights"
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "\U0001F4CA Dashboard":

    st.header("\U0001F4CA VOXINTEL Dashboard")
    total_conversations = len(df)

    if not df.empty:
        total_words = sum(
            len(str(text).split())
            for text in df["transcript"]
        )
        average_words = (
            total_words / total_conversations
            if total_conversations else 0
        )
        total_speakers = df["speaker"].dropna().nunique()
        total_languages = df["language"].dropna().nunique()
        total_events = sum(event_counts(df).values())
    else:
        total_words = 0
        average_words = 0
        total_speakers = 0
        total_languages = 0
        total_events = 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Conversations", total_conversations)
    col2.metric("Total Words", total_words)
    col3.metric("Speakers", total_speakers)
    col4.metric("Languages", total_languages)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Words", round(average_words, 2))
    col2.metric("Total Events", total_events)
    col3.metric("Database Rows", len(df))
    col4.metric("System", "Online")

    st.markdown("---")

    if not df.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Language Distribution")
            st.bar_chart(
                df["language"].fillna("Unknown").value_counts()
            )

        with col2:
            st.subheader("Speaker Activity")
            st.bar_chart(
                df["speaker"].fillna("Unknown").value_counts()
            )

        st.subheader("Event Statistics")
        counts = event_counts(df)
        if counts:
            event_df = pd.DataFrame(
                list(counts.items()),
                columns=["Event", "Count"]
            )
            st.bar_chart(event_df.set_index("Event"))
        else:
            st.info("No events detected yet.")

        st.subheader("Conversation Trend")

        trend_df = df.copy()

        if "timestamp" in trend_df.columns:
            trend_df["timestamp"] = pd.to_datetime(
                trend_df["timestamp"],
                errors="coerce"
            )
            trend_df = trend_df.dropna(subset=["timestamp"])

            if not trend_df.empty:
                daily_trend = (
                    trend_df
                    .set_index("timestamp")
                    .resample("D")
                    .size()
                    .rename("Conversations")
                    .rename_axis("Date")
                    .reset_index()
                )

                import plotly.express as px

                fig = px.line(
                    daily_trend,
                    x="Date",
                    y="Conversations",
                    markers=True,
                    title="Daily Conversation Activity",
                    labels={
                        "Date": "Date",
                        "Conversations": "Number of Conversations"
                    },
                    hover_data={
                        "Date": "|%d %b %Y",
                        "Conversations": True
                    }
                )

                fig.update_traces(
                    line=dict(width=3),
                    marker=dict(size=8),
                    hovertemplate=(
                        "<b>%{x|%d %b %Y}</b><br>"
                        "Conversations: %{y}"
                        "<extra></extra>"
                    )
                )

                tickvals = daily_trend["Date"].iloc[::7].tolist()
                if tickvals[-1] != daily_trend["Date"].iloc[-1]:
                    tickvals.append(daily_trend["Date"].iloc[-1])

                fig.update_layout(
                    height=380,
                    template="plotly_white",
                    hovermode="x unified",
                    margin=dict(l=20, r=20, t=55, b=20),
                    xaxis=dict(
                        title="Date",
                        tickformat="%b %d",
                        tickangle=-30,
                        tickmode="array",
                        tickvals=tickvals,
                        ticktext=[
                            date.strftime("%b %d") for date in tickvals
                        ],
                        range=[
                            daily_trend["Date"].min(),
                            daily_trend["Date"].max()
                        ],
                        showgrid=False
                    ),
                    yaxis=dict(
                        title="Conversations",
                        rangemode="tozero",
                        dtick=1,
                        gridcolor="rgba(128,128,128,0.15)"
                    )
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                    config={"displayModeBar": True}
                )
            else:
                st.info("Not enough timestamp data for a trend chart.")
        else:
            st.info("Timestamp data is not available.")

        st.subheader("Recent Conversations")
        recent = df.head(10)[
            ["id", "timestamp", "transcript", "language", "speaker"]
        ]
        st.dataframe(recent, width="stretch", hide_index=True)

    else:
        st.info(
            "No conversations stored yet. Use Speech-to-Text "
            "to create your first conversation."
        )

    st.markdown("---")
    st.subheader("Data Science Platform")

    cards = [
        ("NLP", "Text analysis, keywords and sentiment"),
        ("Segmentation", "RFM customer intelligence"),
        ("Fraud", "Machine-learning risk detection"),
        ("Forecasting", "Sales trend and prediction"),
        ("EDA", "Data quality and statistics"),
        ("ML", "Classification and regression")
    ]

    cols = st.columns(3)
    for i, (title, description) in enumerate(cards):
        with cols[i % 3]:
            st.info(f"**{title}**\n\n{description}")


# ============================================================
# SPEECH TO TEXT
# ============================================================

elif page == "🎙️ Speech-to-Text":

    st.header("Speech-to-Text")
    st.write("Record your voice and convert it into conversation intelligence.")

    st.info(
        "Select the language you will speak, record your audio, "
        "and then convert it into text."
    )

    st.session_state.setdefault("voxintel_transcript", "")
    st.session_state.setdefault("voxintel_language", "")
    st.session_state.setdefault("voxintel_events", [])
    st.session_state.setdefault("voxintel_saved", False)

    try:
        import speech_recognition as sr
        from audiorecorder import audiorecorder

        language_options = {
            "English (India)": "en-IN",
            "English (US)": "en-US",
            "Telugu": "te-IN",
            "Hindi": "hi-IN",
        }

        selected_language = st.selectbox(
            "Select spoken language",
            options=list(language_options.keys()),
            key="voxintel_recognition_language_choice",
        )
        recognition_language = language_options[selected_language]

        st.caption(
            f"Recognition language: {selected_language} "
            f"({recognition_language})"
        )

        st.subheader("Audio Recording")
        st.caption(
            "Click Start Recording, speak clearly, and click Stop Recording."
        )

        audio = audiorecorder(
            "Start Recording",
            "Stop Recording"
        )

        if len(audio) > 0:
            st.success("Audio recorded successfully!")

            audio_preview = io.BytesIO()
            audio.export(audio_preview, format="wav")
            audio_preview.seek(0)

            st.audio(audio_preview, format="audio/wav")

            if st.button(
                "Convert Speech to Text",
                type="primary",
                key="voxintel_convert_speech"
            ):
                try:
                    recognizer = sr.Recognizer()
                    wav_buffer = io.BytesIO()

                    with st.spinner(
                        f"Converting your speech in {selected_language}..."
                    ):
                        audio.export(wav_buffer, format="wav")
                        wav_buffer.seek(0)

                        with sr.AudioFile(wav_buffer) as source:
                            audio_data = recognizer.record(source)

                        text_result = recognizer.recognize_google(
                            audio_data,
                            language=recognition_language
                        ).strip()

                    if text_result:
                        st.session_state["voxintel_transcript"] = text_result
                        st.session_state["voxintel_language"] = (
                            language_detector.detect_language(text_result)
                        )
                        st.session_state["voxintel_events"] = (
                            event_extractor.extract_events(text_result)
                        )
                        st.session_state["voxintel_saved"] = False

                        st.success("Speech recognized successfully!")
                        st.rerun()
                    else:
                        st.warning(
                            "No speech was detected. Please record again."
                        )

                except sr.UnknownValueError:
                    st.warning(
                        "Speech could not be understood. "
                        "Speak clearly and try recording again."
                    )

                except sr.RequestError as exc:
                    st.error(
                        "Could not connect to Google Speech Recognition. "
                        "Check your internet connection and try again."
                    )
                    st.caption(f"Technical details: {exc}")

                except Exception as exc:
                    st.error(f"Transcription failed: {exc}")

        else:
            st.info(
                "No audio recorded yet. Use the recording controls above "
                "to begin."
            )

        # Display the transcript outside the conversion button.
        text_result = st.session_state["voxintel_transcript"]

        if text_result:
            st.subheader("Transcript")

            st.text_area(
                "Recognized Text",
                value=text_result,
                height=150,
                disabled=True,
                key="voxintel_transcript_display"
            )

            detected_language = st.session_state["voxintel_language"]
            detected_events = st.session_state["voxintel_events"]

            col1, col2 = st.columns(2)
            col1.metric("Language", detected_language)
            col2.metric("Events", len(detected_events))

            if detected_events:
                st.subheader("Detected Events")
                display_event_cards(detected_events)

            if st.session_state["voxintel_saved"]:
                st.success(
                    f"Conversation already saved. "
                    f"Database ID: {st.session_state.get('voxintel_saved_id', 'Unknown')}"
                )

            elif st.button(
                "Save Conversation",
                type="primary",
                key="voxintel_save_conversation"
            ):
                try:
                    speaker = speaker_detector.assign_speaker("voice_001")
                    analysis = SentimentAnalyzer().analyze(text_result)

                    conversation_id = database.save_conversation(
                        transcript=text_result,
                        language=detected_language,
                        speaker=speaker,
                        events=detected_events,
                        sentiment=analysis["sentiment"],
                        polarity=analysis["polarity"],
                        subjectivity=analysis["subjectivity"],
                        confidence=None
                    )

                    st.session_state["voxintel_saved"] = True
                    st.session_state["voxintel_saved_id"] = conversation_id

                    try:
                        memory.add_conversation(
                            speaker=speaker,
                            text=text_result,
                            language=detected_language,
                            events=detected_events
                        )
                    except Exception as memory_error:
                        st.warning(
                            f"Cloud save succeeded, but local memory update failed: {memory_error}"
                        )

                    st.success(
                        f"Conversation saved successfully! Database ID: {conversation_id}"
                    )
                    st.rerun()

                except Exception as exc:
                    st.error(
                        f"Could not save conversation to Supabase: {exc}"
                    )

    except ImportError as exc:
        st.error(f"Speech-to-Text dependency is missing: {exc}")

    except Exception as exc:
        st.error(f"Speech-to-Text error: {exc}")


elif "Conversation Intelligence" in page:

    st.header("🧠 Conversation Intelligence")
    st.write(
        "Analyze saved and new conversations using language detection, "
        "speaker identification and event extraction."
    )

    st.subheader("📂 Analyze Saved Conversations")

    if df.empty:
        st.info("No saved conversations are available in the database.")
    else:
        conversation_options = list(range(len(df)))

        selected_index = st.selectbox(
            "Select a saved conversation",
            options=conversation_options,
            format_func=lambda index: (
                f"ID {df.iloc[index]['id']} | "
                f"{str(df.iloc[index]['transcript'])[:75]}"
            ),
            key="ci_saved_conversation"
        )

        selected_row = df.iloc[selected_index]
        saved_transcript = str(
            selected_row.get("transcript", "")
        ).strip()

        with st.expander("📄 View Saved Transcript", expanded=True):
            st.write(saved_transcript)

        if st.button(
            "🔍 Analyze Selected Conversation",
            type="primary",
            key="ci_analyze_saved"
        ):
            if not saved_transcript or saved_transcript.lower() == "nan":
                st.warning("This conversation has no transcript.")
            else:
                try:
                    language = language_detector.detect_language(
                        saved_transcript
                    )

                    stored_speaker = selected_row.get("speaker")

                    if (
                        stored_speaker is None
                        or str(stored_speaker).strip().lower()
                        in ("", "nan", "none")
                    ):
                        speaker = speaker_detector.assign_speaker(
                            "voice_001"
                        )
                    else:
                        speaker = str(stored_speaker)

                    events = event_extractor.extract_events(
                        saved_transcript
                    )

                    st.success("Saved conversation analyzed successfully!")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("🌐 Language", language)
                    col2.metric("🎙️ Speaker", speaker)
                    col3.metric(
                        "📌 Events",
                        len(events) if events is not None else 0
                    )

                    st.subheader("📝 Transcript")
                    st.write(saved_transcript)

                    st.subheader("⚡ Extracted Events")

                    display_event_cards(events)

                except Exception as exc:
                    st.error(f"Conversation analysis failed: {exc}")

    st.divider()
    st.subheader("✨ Analyze New Text")

    text_input = st.text_area(
        "Enter conversation text",
        height=150,
        placeholder=(
            "Example: We have a meeting tomorrow at 10 AM "
            "in Hyderabad. Please remind me to call Rahul."
        ),
        key="ci_manual_text"
    )

    if st.button(
        "✨ Analyze New Text",
        key="ci_analyze_manual"
    ):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            try:
                language = language_detector.detect_language(text_input)
                speaker = speaker_detector.assign_speaker("voice_001")
                events = event_extractor.extract_events(text_input)

                st.success("Conversation analyzed successfully!")

                col1, col2, col3 = st.columns(3)
                col1.metric("🌐 Language", language)
                col2.metric("🎙️ Speaker", speaker)
                col3.metric(
                    "📌 Events",
                    len(events) if events is not None else 0
                )

                st.subheader("📝 Transcript")
                st.write(text_input)

                st.subheader("⚡ Extracted Events")

                display_event_cards(events)

            except Exception as exc:
                st.error(f"Conversation analysis failed: {exc}")


elif page == "\U0001F60A Sentiment & NLP":

    st.header("😊 Sentiment & NLP")
    st.write(
        "Analyze conversation text using NLP, sentiment analysis, "
        "keywords and text statistics."
    )

    text = st.text_area(
        "Enter text for NLP analysis",
        height=180,
        placeholder="Example: The customer is very happy with our service."
    )

    if st.button("🧠 Run NLP Analysis", type="primary"):

        if not text.strip():
            st.warning("Please enter text.")
        else:
            analyzer = TextAnalyzer()
            sentiment = SentimentAnalyzer()
            advanced = AdvancedNLP()

            result = sentiment.analyze(text)
            stats = analyzer.statistics(text)
            keywords = analyzer.keywords(text, 10)
            summary = advanced.text_summary(text)

            st.subheader("😊 Sentiment")
            c1, c2, c3 = st.columns(3)
            c1.metric("Sentiment", result["sentiment"])
            c2.metric("Polarity", round(result["polarity"], 3))
            c3.metric("Subjectivity", round(result["subjectivity"], 3))

            st.subheader("📊 Text Statistics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Characters", stats["characters"])
            c2.metric("Words", stats["words"])
            c3.metric("Unique Words", stats["unique_words"])
            c4.metric("Sentences", stats["sentences"])

            st.subheader("🔑 Top Keywords")
            keyword_df = pd.DataFrame(
                keywords,
                columns=["Keyword", "Frequency"]
            )
            st.dataframe(
                keyword_df,
                width="stretch",
                hide_index=True
            )

            st.subheader("🧠 Advanced NLP Summary")
            st.json(summary)


    # ========================================================
    # HUMAN SENTIMENT REVIEW
    # ========================================================

    st.divider()
    st.header("Human Review")

    if df.empty:
        st.info("No conversations are available for review.")
    else:
        review_data = df.copy()
        review_data = review_data.dropna(subset=["id", "transcript"])
        review_data["reviewed_sentiment"] = (
            review_data["reviewed_sentiment"].fillna("").astype(str)
        )
        review_data["sentiment"] = (
            review_data["sentiment"].fillna("").astype(str)
        )

        labels = ["Negative", "Neutral", "Positive"]

        reviewed_data = review_data[
            review_data["reviewed_sentiment"].isin(labels)
        ]
        pending_data = review_data[
            ~review_data["reviewed_sentiment"].isin(labels)
        ]

        total_count = len(review_data)
        reviewed_count = len(reviewed_data)
        pending_count = len(pending_data)

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Total Conversations", total_count)
        metric2.metric("Human Reviewed", reviewed_count)
        metric3.metric("Pending Reviews", pending_count)

        st.subheader("Review Conversations")

        review_data["_reviewed"] = (
            review_data["reviewed_sentiment"].isin(labels)
        )
        review_data = review_data.sort_values(
            ["_reviewed", "id"],
            ascending=[True, False]
        )

        conversation_ids = [
            int(value) for value in review_data["id"].tolist()
        ]

        review_labels = {}
        for _, row in review_data.iterrows():
            conversation_id = int(row["id"])
            transcript_preview = str(row["transcript"]).replace("\n", " ")
            review_labels[conversation_id] = (
                f"ID {conversation_id} | {transcript_preview[:65]}"
            )

        selected_id = st.selectbox(
            "Select a conversation",
            options=conversation_ids,
            format_func=lambda value: review_labels[value],
            key="human_review_conversation"
        )

        selected_row = review_data[
            review_data["id"].astype(int) == selected_id
        ].iloc[0]

        st.markdown("**Conversation transcript**")
        st.write(selected_row["transcript"])

        st.write(
            "AI-predicted sentiment:",
            selected_row["sentiment"] or "Not available"
        )

        current_review = selected_row["reviewed_sentiment"]
        default_index = (
            labels.index(current_review)
            if current_review in labels else 0
        )

        reviewed_sentiment = st.selectbox(
            "Your human sentiment classification",
            labels,
            index=default_index,
            key=f"human_review_label_{selected_id}"
        )

        current_notes = selected_row["review_notes"]
        if pd.isna(current_notes):
            current_notes = ""

        review_notes = st.text_area(
            "Review notes",
            value=str(current_notes),
            placeholder="Explain why you selected this sentiment...",
            key=f"human_review_notes_{selected_id}"
        )

        if st.button("Save Human Review", type="primary"):
            try:
                database.save_review(
                    conversation_id=selected_id,
                    reviewed_sentiment=reviewed_sentiment,
                    review_notes=review_notes.strip()
                )
                st.success("Human review saved successfully to Supabase.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save review: {exc}")

        # ====================================================
        # SENTIMENT MODEL EVALUATION
        # ====================================================

        st.divider()
        st.header("Sentiment Model Evaluation")

        evaluation_data = review_data[
            review_data["reviewed_sentiment"].isin(labels)
            & review_data["sentiment"].isin(labels)
        ].copy()

        if evaluation_data.empty:
            st.info(
                "Model evaluation requires conversations with both "
                "an AI prediction and a human-reviewed sentiment."
            )
        else:
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                confusion_matrix,
                classification_report
            )

            actual = evaluation_data["reviewed_sentiment"]
            predicted = evaluation_data["sentiment"]

            accuracy = accuracy_score(actual, predicted)
            precision = precision_score(
                actual, predicted, labels=labels,
                average="macro", zero_division=0
            )
            recall = recall_score(
                actual, predicted, labels=labels,
                average="macro", zero_division=0
            )
            f1 = f1_score(
                actual, predicted, labels=labels,
                average="macro", zero_division=0
            )

            st.caption(
                f"Evaluation uses {len(evaluation_data)} conversations "
                "that have both AI predictions and human labels."
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{accuracy:.1%}")
            m2.metric("Macro Precision", f"{precision:.1%}")
            m3.metric("Macro Recall", f"{recall:.1%}")
            m4.metric("Macro F1-score", f"{f1:.1%}")

            st.subheader("Confusion Matrix")

            matrix = confusion_matrix(
                actual,
                predicted,
                labels=labels
            )

            matrix_df = pd.DataFrame(
                matrix,
                index=[f"Actual {label}" for label in labels],
                columns=[f"AI {label}" for label in labels]
            )

            st.dataframe(
                matrix_df,
                width="stretch"
            )

            st.subheader("Classification Report")

            report = classification_report(
                actual,
                predicted,
                labels=labels,
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(report).T

            st.dataframe(
                report_df.round(3),
                width="stretch"
            )

            st.caption(
                "Human labels are treated as reference labels. "
                "AI predictions are not modified by this evaluation."
            )



# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

elif page == "\U0001F465 Customer Segmentation":

    st.header("👥 Customer Segmentation")
    st.write(
        "Segment customers using Recency, Frequency and Monetary (RFM) analysis."
    )

    uploaded = st.file_uploader(
        "Upload customer transaction CSV",
        type=["csv"],
        key="segmentation_csv"
    )

    if uploaded:
        data = pd.read_csv(uploaded)
        st.write("Dataset preview")
        st.dataframe(data.head(), width="stretch")

        columns = data.columns.tolist()

        if len(columns) >= 3:
            c1, c2, c3 = st.columns(3)

            customer_col = c1.selectbox(
                "Customer column",
                columns,
                key="seg_customer"
            )
            date_col = c2.selectbox(
                "Date column",
                columns,
                key="seg_date"
            )
            value_col = c3.selectbox(
                "Value / Sales column",
                columns,
                key="seg_value"
            )

            clusters = st.slider(
                "Number of clusters",
                min_value=2,
                max_value=6,
                value=4
            )

            if st.button(
                "👥 Run Customer Segmentation",
                type="primary"
            ):
                try:
                    segmenter = CustomerSegmentation(data)
                    rfm = segmenter.run(
                        customer_col,
                        date_col,
                        value_col,
                        clusters
                    )

                    st.subheader("👥 Customer Segments")
                    st.dataframe(
                        rfm,
                        width="stretch",
                        hide_index=True
                    )

                    summary = segmenter.segment_summary(rfm)

                    st.subheader("📊 Segment Summary")
                    st.dataframe(
                        summary,
                        width="stretch",
                        hide_index=True
                    )

                    if not summary.empty:
                        st.bar_chart(
                            summary.set_index("Segment")["Customers"]
                        )

                except Exception as e:
                    st.error(f"Segmentation error: {e}")

    else:
        st.info(
            "Upload a CSV containing customer, date and transaction value "
            "columns to run RFM segmentation."
        )


# ============================================================
# FRAUD DETECTION
# ============================================================

elif page == "\U0001F6A8 Fraud Detection":

    st.header("🚨 Fraud Detection")
    st.write(
        "Train a Random Forest model to identify potentially fraudulent transactions."
    )

    uploaded = st.file_uploader(
        "Upload fraud transaction CSV",
        type=["csv"],
        key="fraud_csv"
    )

    if uploaded:
        data = pd.read_csv(uploaded)

        st.dataframe(
            data.head(),
            width="stretch"
        )

        target = st.selectbox(
            "Select fraud target column",
            data.columns.tolist(),
            key="fraud_target"
        )

        if st.button(
            "🚨 Train Fraud Model",
            type="primary"
        ):
            try:
                detector = FraudDetector(data)
                metrics = detector.train_model(target)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                c2.metric("Precision", f"{metrics['precision']:.2%}")
                c3.metric("Recall", f"{metrics['recall']:.2%}")
                c4.metric("F1 Score", f"{metrics['f1']:.2%}")

                st.subheader("📊 Feature Importance")
                importance = detector.feature_importance()
                st.dataframe(
                    importance,
                    width="stretch",
                    hide_index=True
                )
                st.bar_chart(
                    importance.set_index("Feature")["Importance"]
                )

                st.session_state["fraud_detector"] = detector

            except Exception as e:
                st.error(f"Fraud model error: {e}")

    else:
        st.info(
            "Upload a labelled transaction CSV to train the fraud detection model."
        )


# ============================================================
# SALES FORECASTING
# ============================================================

elif page == "\U0001F4C8 Sales Forecasting":

    st.header("📈 Sales Forecasting")
    st.write(
        "Analyze historical sales trends, moving averages and future sales."
    )

    uploaded = st.file_uploader(
        "Upload sales CSV",
        type=["csv"],
        key="forecast_csv"
    )

    if uploaded:
        data = pd.read_csv(uploaded)
        st.dataframe(data.head(), width="stretch")

        columns = data.columns.tolist()

        if len(columns) >= 2:
            c1, c2 = st.columns(2)

            date_col = c1.selectbox(
                "Date column",
                columns,
                key="forecast_date"
            )

            sales_col = c2.selectbox(
                "Sales column",
                columns,
                key="forecast_sales"
            )

            periods = st.slider(
                "Forecast days",
                min_value=1,
                max_value=30,
                value=7
            )

            if st.button(
                "📈 Generate Forecast",
                type="primary"
            ):
                try:
                    forecaster = SalesForecaster(data)

                    prepared = forecaster.prepare_data(
                        date_col,
                        sales_col
                    )

                    trend = forecaster.trend(
                        date_col,
                        sales_col
                    )

                    moving = forecaster.moving_average(
                        date_col,
                        sales_col,
                        3
                    )

                    forecast = forecaster.forecast(
                        date_col,
                        sales_col,
                        periods
                    )

                    summary = forecaster.summary(
                        date_col,
                        sales_col,
                        periods
                    )

                    c1, c2, c3 = st.columns(3)
                    c1.metric(
                        "Historical Sales",
                        f"{summary['historical_sales']:.2f}"
                    )
                    c2.metric(
                        "Forecast Sales",
                        f"{summary['forecast_sales']:.2f}"
                    )
                    c3.metric(
                        "Trend",
                        trend["direction"]
                    )

                    st.subheader("📊 Historical Sales")
                    historical_chart = prepared.set_index(date_col)[
                        [sales_col]
                    ]
                    st.line_chart(historical_chart)

                    st.subheader("📉 Moving Average")
                    moving_chart = moving.set_index(date_col)[
                        [sales_col, "Moving Average"]
                    ]
                    st.line_chart(moving_chart)

                    st.subheader("🔮 Future Sales Forecast")
                    st.dataframe(
                        forecast,
                        width="stretch",
                        hide_index=True
                    )

                    st.line_chart(
                        forecast.set_index("Date")[
                            ["Forecast Sales"]
                        ]
                    )

                    st.success(
                        f"Forecast completed for the next {periods} days."
                    )

                except Exception as e:
                    st.error(f"Forecasting error: {e}")

    else:
        st.info(
            "Upload a CSV containing a date column and a sales/revenue column."
        )


# ============================================================
# BUSINESS ANALYTICS
# ============================================================

elif page == "\U0001F4BC Business Analytics":

    st.header("💼 Business Analytics")
    st.write(
        "Generate KPIs, group analysis, customer analysis and business insights."
    )

    uploaded = st.file_uploader(
        "Upload business CSV",
        type=["csv"],
        key="business_csv"
    )

    if uploaded:
        data = pd.read_csv(uploaded)

        st.dataframe(
            data.head(),
            width="stretch"
        )

        numeric_columns = data.select_dtypes(
            include="number"
        ).columns.tolist()

        all_columns = data.columns.tolist()

        if numeric_columns:
            value_col = st.selectbox(
                "Select numeric value column",
                numeric_columns,
                key="business_value"
            )

            if st.button(
                "📊 Analyze Business Data",
                type="primary"
            ):
                try:
                    business = BusinessAnalytics(data)
                    summary = business.summary()
                    kpi = business.calculate_kpi(value_col)
                    insights = business.generate_insights(value_col)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Rows", summary["rows"])
                    c2.metric("Columns", summary["columns"])
                    c3.metric(
                        "Numeric Columns",
                        summary["total_numeric_columns"]
                    )

                    st.subheader("KPI Analysis")
                    st.json(kpi)

                    st.subheader("Business Insights")
                    for insight in insights:
                        st.info(insight)

                    st.subheader("Outlier Analysis")
                    outliers = business.detect_outliers(value_col)
                    st.write(
                        f"Outliers detected: **{outliers['count']}**"
                    )
                    st.write(
                        f"Lower bound: **{outliers['lower_bound']:.2f}**"
                    )
                    st.write(
                        f"Upper bound: **{outliers['upper_bound']:.2f}**"
                    )

                    if outliers["count"] > 0:
                        st.dataframe(
                            outliers["outliers"],
                            width="stretch",
                            hide_index=True
                        )

                except Exception as e:
                    st.error(f"Business analytics error: {e}")

        else:
            st.warning("No numeric columns found in this dataset.")

    else:
        st.info("Upload a business dataset CSV to begin analysis.")


# ============================================================
# EDA & STATISTICS
# ============================================================

elif page == "📋 EDA & Statistics":

    st.header("📋 EDA & Statistics")
    st.write(
        "Explore data quality, descriptive statistics, correlations and distributions."
    )

    uploaded = st.file_uploader(
        "Upload CSV for EDA",
        type=["csv"],
        key="eda_csv"
    )

    if uploaded:
        data = pd.read_csv(uploaded)

        eda = EDAAnalyzer(data)
        stats = StatisticsAnalyzer(data)

        summary = eda.summary()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", summary["rows"])
        c2.metric("Columns", summary["columns"])
        c3.metric("Missing Values", summary["missing_values"])
        c4.metric("Duplicates", summary["duplicate_rows"])

        st.subheader("📋 Dataset Preview")
        st.dataframe(
            data.head(20),
            width="stretch",
            hide_index=True
        )

        st.subheader("🧾 Data Types")
        st.dataframe(
            eda.data_types().rename("Data Type").to_frame(),
            width="stretch"
        )

        st.subheader("📊 Descriptive Statistics")
        st.dataframe(
            stats.numeric_summary(),
            width="stretch"
        )

        st.subheader("🔗 Correlation Matrix")
        correlation = eda.correlation()
        if not correlation.empty:
            st.dataframe(
                correlation,
                width="stretch"
            )
        else:
            st.info("No numeric columns available.")

        numeric_columns = eda.numerical_columns()
        categorical_columns = eda.categorical_columns()

        if numeric_columns:
            st.subheader("📊 Numeric Visualization")
            selected_numeric = st.selectbox(
                "Select numeric column",
                numeric_columns,
                key="eda_numeric"
            )

            fig = VisualizationEngine.histogram(
                data,
                selected_numeric
            )
            st.plotly_chart(fig, width="stretch")

        if categorical_columns:
            st.subheader("📊 Categorical Visualization")
            selected_category = st.selectbox(
                "Select categorical column",
                categorical_columns,
                key="eda_category"
            )

            fig = VisualizationEngine.bar_chart(
                data,
                selected_category
            )
            st.plotly_chart(fig, width="stretch")

    else:
        st.info("Upload a CSV dataset to perform EDA.")


# ============================================================
# MACHINE LEARNING
# ============================================================

elif page == "📊 Machine Learning":

    st.header("📊 Machine Learning")
    st.write(
        "Train classification or regression models on uploaded datasets."
    )

    uploaded = st.file_uploader(
        "Upload ML dataset CSV",
        type=["csv"],
        key="ml_csv"
    )

    if uploaded:
        data = pd.read_csv(uploaded)

        st.dataframe(
            data.head(),
            width="stretch"
        )

        target = st.selectbox(
            "Select target column",
            data.columns.tolist(),
            key="ml_target"
        )

        task = st.radio(
            "Machine Learning Task",
            ["Classification", "Regression"],
            horizontal=True
        )

        if task == "Classification":
            model_name = st.selectbox(
                "Select Classification Model",
                [
                    "Random Forest",
                    "Logistic Regression",
                    "Decision Tree"
                ]
            )
        else:
            model_name = st.selectbox(
                "Select Regression Model",
                [
                    "Random Forest",
                    "Linear Regression"
                ]
            )

        if st.button(
            "🤖 Train Machine Learning Model",
            type="primary"
        ):
            try:
                ml = MLAnalyzer(data)

                if task == "Classification":
                    result = ml.classification(
                        target,
                        model_name
                    )

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(
                        "Accuracy",
                        f"{result['accuracy']:.2%}"
                    )
                    c2.metric(
                        "Precision",
                        f"{result['precision']:.2%}"
                    )
                    c3.metric(
                        "Recall",
                        f"{result['recall']:.2%}"
                    )
                    c4.metric(
                        "F1 Score",
                        f"{result['f1']:.2%}"
                    )

                else:
                    result = ml.regression(
                        target,
                        model_name
                    )

                    c1, c2, c3 = st.columns(3)
                    c1.metric(
                        "RÃ‚Â²",
                        f"{result['r2']:.4f}"
                    )
                    c2.metric(
                        "MAE",
                        f"{result['mae']:.4f}"
                    )
                    c3.metric(
                        "RMSE",
                        f"{result['rmse']:.4f}"
                    )

                st.success(
                    f"{model_name} trained successfully."
                )

            except Exception as e:
                st.error(f"Machine learning error: {e}")

    else:
        st.info("Upload a dataset CSV to train a machine-learning model.")


# ============================================================
# KNOWLEDGE GRAPH
# ============================================================

elif page == "\U0001F578 Knowledge Graph":

    st.header("🕸️ Interactive Knowledge Graph")
    st.write(
        "Explore relationships between conversations, speakers, "
        "languages, events and transcripts."
    )

    try:
        from streamlit_agraph import agraph, Node, Edge, Config

        if df.empty:
            st.info("No conversation data available.")
        else:
            nodes = []
            edges = []
            node_ids = set()
            edge_ids = set()

            def add_node(node_id, label, size=25):
                if node_id not in node_ids:
                    nodes.append(
                        Node(
                            id=node_id,
                            label=label,
                            size=size,
                            shape="dot"
                        )
                    )
                    node_ids.add(node_id)

            def add_edge(source, target, label):
                edge_key = f"{source}__{target}__{label}"
                if edge_key not in edge_ids:
                    edges.append(
                        Edge(
                            source=source,
                            target=target,
                            label=label
                        )
                    )
                    edge_ids.add(edge_key)

            for _, row in df.iterrows():
                conversation_id = str(row["id"])
                conversation_node = f"conversation_{conversation_id}"

                add_node(
                    conversation_node,
                    f"Conversation {conversation_id}",
                    35
                )

                speaker = row["speaker"]
                if pd.notna(speaker):
                    speaker_node = (
                        "speaker_" +
                        str(speaker).replace(" ", "_")
                    )
                    add_node(
                        speaker_node,
                        str(speaker),
                        30
                    )
                    add_edge(
                        speaker_node,
                        conversation_node,
                        "SPOKE_IN"
                    )

                language = row["language"]
                if pd.notna(language):
                    language_node = (
                        "language_" +
                        str(language).replace(" ", "_")
                    )
                    add_node(
                        language_node,
                        str(language),
                        30
                    )
                    add_edge(
                        conversation_node,
                        language_node,
                        "HAS_LANGUAGE"
                    )

                transcript = row["transcript"]
                if pd.notna(transcript):
                    transcript_node = f"transcript_{conversation_id}"
                    short_text = str(transcript)
                    if len(short_text) > 35:
                        short_text = short_text[:35] + "..."
                    add_node(
                        transcript_node,
                        short_text,
                        20
                    )
                    add_edge(
                        conversation_node,
                        transcript_node,
                        "HAS_TRANSCRIPT"
                    )

                for index, event in enumerate(
                    parse_events(row["events"])
                ):
                    if isinstance(event, dict):
                        event_type = event.get("type", "Unknown")
                    else:
                        event_type = str(event)

                    event_node = (
                        f"event_{conversation_id}_{index}"
                    )
                    add_node(
                        event_node,
                        event_type,
                        25
                    )
                    add_edge(
                        conversation_node,
                        event_node,
                        "HAS_EVENT"
                    )

            config = Config(
                width="100%",
                height=650,
                directed=True,
                physics=True,
                hierarchical=False,
                nodeHighlightBehavior=True,
                highlightColor="#F7A072",
                collapsible=False
            )

            agraph(
                nodes=nodes,
                edges=edges,
                config=config
            )

            col1, col2 = st.columns(2)
            col1.metric("Nodes", len(nodes))
            col2.metric("Relationships", len(edges))

    except ImportError:
        st.error("streamlit-agraph is not installed.")
        st.code("pip install streamlit-agraph")
    except Exception as e:
        st.error(f"Knowledge Graph error: {e}")


# ============================================================
# CONVERSATION DATABASE
# ============================================================

elif page == "\U0001F5C4 Conversation Database":

    st.header("Conversation Database")

    if df.empty:
        st.info("No conversations stored.")
    else:
        st.write(f"Total conversations: **{len(df)}**")

        st.dataframe(
            df,
            width="stretch",
            hide_index=True
        )

        st.markdown("---")
        st.subheader("Export Database")

        csv_data = df.to_csv(index=False)

        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="voxintel_conversations.csv",
            mime="text/csv"
        )


# ============================================================
# CONVERSATION ANALYTICS
# ============================================================

elif page == "\U0001F4CA Analytics":

    st.header("Conversation Analytics")

    if df.empty:
        st.info("No conversation data available.")
    else:
        st.subheader("Language Analytics")

        language_distribution = (
            df["language"]
            .fillna("Unknown")
            .value_counts()
        )

        col1, col2 = st.columns(2)

        with col1:
            st.bar_chart(language_distribution)

        with col2:
            st.dataframe(
                language_distribution
                .rename("Conversations")
                .to_frame(),
                width="stretch"
            )

        st.markdown("---")
        st.subheader("📊 Speaker Analytics")

        speaker_distribution = (
            df["speaker"]
            .fillna("Unknown")
            .value_counts()
        )

        col1, col2 = st.columns(2)

        with col1:
            st.bar_chart(speaker_distribution)

        with col2:
            st.dataframe(
                speaker_distribution
                .rename("Conversations")
                .to_frame(),
                width="stretch"
            )

        st.markdown("---")
        st.subheader("📏 Conversation Length")

        word_counts = [
            len(str(text).split())
            for text in df["transcript"]
        ]

        word_df = pd.DataFrame({
            "Conversation": df["id"],
            "Words": word_counts
        })

        st.bar_chart(
            word_df.set_index("Conversation")
        )

        st.markdown("---")
        st.subheader("📅 Event Analytics")

        counts = event_counts(df)

        if counts:
            event_df = pd.DataFrame(
                list(counts.items()),
                columns=["Event", "Count"]
            )

            st.bar_chart(
                event_df.set_index("Event")
            )

            st.dataframe(
                event_df,
                width="stretch",
                hide_index=True
            )
        else:
            st.info("No events available.")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "VOXINTEL - AI-Powered Multilingual Conversation & "
    "Customer Intelligence Platform"
)

st.caption(
    "Speech > Text > Language > Speaker > Events > "
    "Memory > Database > NLP > Customer Intelligence > "
    "Data Science > Business Insights"
)
