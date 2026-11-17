import re


class EventExtractor:

    LOCATIONS = [
        "hyderabad", "visakhapatnam", "vizag",
        "bangalore", "bengaluru", "chennai",
        "mumbai", "delhi", "pune"
    ]

    def extract_events(self, text):

        events = []

        if not text or not text.strip():
            return events

        text_lower = text.lower().strip()

        # Extract time
        time_match = re.search(
            r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?\s*m\.?|p\.?\s*m\.?|am|pm)\b",
            text_lower
        )
        time_value = time_match.group().strip() if time_match else None

        # Extract numeric or relative date
        date_patterns = [
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
            r"\b(today|tomorrow|yesterday|tonight)\b"
        ]

        date_value = None
        for pattern in date_patterns:
            match = re.search(pattern, text_lower)
            if match:
                date_value = match.group().strip()
                break

        # Extract known locations
        location_value = None
        for location in self.LOCATIONS:
            if re.search(r"\b" + re.escape(location) + r"\b", text_lower):
                location_value = location.title()
                break

        # Extract person following call/phone verbs
        person_match = re.search(
            r"\b(?:call|phone)\s+(?:mr\.?\s+|mrs\.?\s+|ms\.?\s+)?"
            r"([A-Z][a-z]+)\b",
            text
        )
        person_value = person_match.group(1) if person_match else None

        # Extract reminder action
        action_match = re.search(
            r"\b(?:remind(?:\s+me)?\s+to|remember\s+to)\s+"
            r"([^.!?]+)",
            text,
            re.IGNORECASE
        )
        action_value = action_match.group(1).strip() if action_match else None

        if action_value:
            action_value = re.sub(r"\s+", " ", action_value)
            action_value = action_value[0].upper() + action_value[1:]

        # MEETING
        if any(word in text_lower for word in [
            "meeting", "conference", "appointment"
        ]):
            event = {
                "type": "Meeting",
                "text": text
            }
            if date_value:
                event["date"] = date_value
            if time_value:
                event["time"] = time_value
            if location_value:
                event["location"] = location_value
            events.append(event)

        # REMINDER
        if any(word in text_lower for word in [
            "remind", "remember", "reminder"
        ]):
            event = {
                "type": "Reminder",
                "text": text
            }
            if action_value:
                event["action"] = action_value
            if person_value:
                event["person"] = person_value
            events.append(event)

        # CALL
        if any(word in text_lower for word in [
            "call", "phone", "telephone"
        ]):
            event = {
                "type": "Call",
                "text": text
            }
            if person_value:
                event["person"] = person_value
            events.append(event)

        # TASK
        if any(word in text_lower for word in [
            "todo", "to-do", "task", "complete",
            "finish", "submit", "prepare"
        ]):
            events.append({
                "type": "Task",
                "text": text
            })

        # QUESTION
        question_words = [
            "what", "why", "when", "where",
            "who", "how", "can you", "could you", "will you"
        ]

        if (
            "?" in text
            or any(text_lower.startswith(word) for word in question_words)
        ):
            events.append({
                "type": "Question",
                "text": text
            })

        # TIME
        if time_value:
            events.append({
                "type": "Time",
                "value": time_value
            })

        # DATE
        if date_value:
            events.append({
                "type": "Date",
                "value": date_value
            })

        # PHONE
        phone_match = re.search(
            r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)",
            text
        )
        if phone_match:
            events.append({
                "type": "Phone",
                "value": phone_match.group()
            })

        # MONEY
        money_match = re.search(
            r"(?:\u20b9|rs\.?|inr)\s?[\d,]+(?:\.\d{1,2})?",
            text,
            re.IGNORECASE
        )
        if money_match:
            events.append({
                "type": "Amount",
                "value": money_match.group()
            })

        # LOCATION
        if location_value:
            events.append({
                "type": "Location",
                "value": location_value
            })

        return events


if __name__ == "__main__":

    extractor = EventExtractor()

    test_sentence = (
        "We have a meeting tomorrow at 10 AM in Hyderabad. "
        "Please remind me to call Rahul."
    )

    print("VOXINTEL Structured Event Extraction Test")
    print("=" * 55)

    for event in extractor.extract_events(test_sentence):
        print(event)
