from datetime import datetime
import time

import psycopg
import streamlit as st


class ConversationDatabase:

    def __init__(self):
        # Supabase schema already exists.
        # No local SQLite database is created or modified.
        pass

    def connect(self):
        """Connect securely to Supabase with retry handling."""

        try:
            config = st.secrets["supabase"]
        except Exception as exc:
            raise RuntimeError(
                "Supabase credentials are missing. "
                "Configure the [supabase] section in Streamlit Secrets."
            ) from exc

        required = [
            "host",
            "port",
            "database",
            "user",
            "password"
        ]

        missing = [
            key for key in required
            if not config.get(key)
        ]

        if missing:
            raise RuntimeError(
                "Missing Supabase configuration: "
                + ", ".join(missing)
            )

        for attempt in range(1, 4):
            try:
                return psycopg.connect(
                    host=config["host"],
                    port=int(config["port"]),
                    dbname=config["database"],
                    user=config["user"],
                    password=config["password"],
                    sslmode=config.get("sslmode", "require"),
                    connect_timeout=15,
                    prepare_threshold=None
                )

            except psycopg.OperationalError:
                if attempt == 3:
                    raise

                time.sleep(2)

    def save_conversation(
        self,
        transcript,
        language,
        speaker,
        events,
        sentiment=None,
        polarity=None,
        subjectivity=None,
        confidence=None
    ):
        """Save a new conversation to Supabase."""

        with self.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO public.conversations
                    (
                        timestamp,
                        transcript,
                        language,
                        speaker,
                        events,
                        sentiment,
                        polarity,
                        subjectivity,
                        confidence
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        datetime.now().isoformat(),
                        transcript,
                        language,
                        speaker,
                        str(events),
                        sentiment,
                        polarity,
                        subjectivity,
                        confidence
                    )
                )

                return cursor.fetchone()[0]

    def get_all_conversations(self):
        """Retrieve all conversation and human-review fields."""

        with self.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        id,
                        timestamp,
                        transcript,
                        language,
                        speaker,
                        events,
                        sentiment,
                        polarity,
                        subjectivity,
                        confidence,
                        reviewed_sentiment,
                        review_notes,
                        reviewed_at
                    FROM public.conversations
                    ORDER BY id DESC
                    """
                )

                return cursor.fetchall()

    def get_conversation_count(self):
        """Return the total number of cloud conversations."""

        with self.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM public.conversations
                    """
                )

                return cursor.fetchone()[0]

    def save_review(
        self,
        conversation_id,
        reviewed_sentiment,
        review_notes=""
    ):
        """Save a human sentiment review without changing AI results."""

        valid_labels = ["Positive", "Neutral", "Negative"]

        if reviewed_sentiment not in valid_labels:
            raise ValueError(
                "Review label must be Positive, Neutral or Negative."
            )

        with self.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE public.conversations
                    SET
                        reviewed_sentiment = %s,
                        review_notes = %s,
                        reviewed_at = %s
                    WHERE id = %s
                    RETURNING id
                    """,
                    (
                        reviewed_sentiment,
                        review_notes,
                        datetime.now().isoformat(),
                        conversation_id
                    )
                )

                updated = cursor.fetchone()

                if updated is None:
                    raise ValueError(
                        f"Conversation ID {conversation_id} was not found."
                    )

                return updated[0]
