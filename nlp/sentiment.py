from textblob import TextBlob


class SentimentAnalyzer:

    TELUGU_POSITIVE = [
        "బాగుంది", "మంచిది", "అద్భుతం", "సంతోషం",
        "ప్రేమ", "చాలా బాగుంది", "ఉత్తమం"
    ]

    TELUGU_NEGATIVE = [
        "చెడ్డది", "బాగాలేదు", "దారుణం", "చెడు",
        "బాధ", "నిరాశ", "చాలా చెడ్డది"
    ]

    def analyze(self, text):

        text = str(text).strip()

        positive_matches = sum(
            word in text for word in self.TELUGU_POSITIVE
        )
        negative_matches = sum(
            word in text for word in self.TELUGU_NEGATIVE
        )

        if positive_matches > negative_matches:
            return {
                "sentiment": "Positive",
                "polarity": 0.5,
                "subjectivity": 0.5
            }

        if negative_matches > positive_matches:
            return {
                "sentiment": "Negative",
                "polarity": -0.5,
                "subjectivity": 0.5
            }

        result = TextBlob(text).sentiment
        polarity = result.polarity
        subjectivity = result.subjectivity

        if polarity > 0.1:
            sentiment = "Positive"
        elif polarity < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return {
            "sentiment": sentiment,
            "polarity": polarity,
            "subjectivity": subjectivity
        }
