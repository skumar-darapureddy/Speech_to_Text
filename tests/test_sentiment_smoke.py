import unittest
from nlp.sentiment import SentimentAnalyzer
from voxintel.language.language_detection import LanguageDetector

class SmokeTests(unittest.TestCase):
    def test_sentiment(self):
        model = SentimentAnalyzer()
        cases = [
            ("Excellent service.", "Positive"),
            ("This service is terrible.", "Negative"),
            ("The office opens at nine tomorrow morning.", "Neutral"),
            ("my name is Santosh", "Neutral"),
            ("\u0c08 \u0c38\u0c47\u0c35 \u0c1a\u0c3e\u0c32\u0c3e \u0c2c\u0c3e\u0c17\u0c41\u0c02\u0c26\u0c3f", "Positive"),
            ("\u0c08 \u0c38\u0c47\u0c35 \u0c1a\u0c3e\u0c32\u0c3e \u0c1a\u0c46\u0c21\u0c4d\u0c21\u0c26\u0c3f", "Negative"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(model.analyze(text)["sentiment"], expected)

    def test_language_detection(self):
        detector = LanguageDetector()
        self.assertEqual(detector.detect_language("Hello, how are you?"), "English")
        self.assertEqual(detector.detect_language(
            "\u0c28\u0c3e \u0c2a\u0c47\u0c30\u0c41 \u0c38\u0c02\u0c24\u0c4b\u0c37\u0c4d"
        ), "Telugu")

if __name__ == "__main__":
    unittest.main()
