import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from voxintel.memory.conversation_memory import ConversationMemory
from voxintel.database.conversation_database import ConversationDatabase
from voxintel.knowledge_graph.knowledge_graph import KnowledgeGraph


class StorageIntegrationTests(unittest.TestCase):

    def test_memory_persistence(self):
        with tempfile.TemporaryDirectory() as folder:
            memory_path = Path(folder) / "test_memory.json"
            memory = ConversationMemory(file_path=str(memory_path))

            memory.add_conversation(
                speaker="Speaker 1",
                text="Meeting tomorrow at 10 AM",
                language="English",
                events=[{"type": "Meeting", "text": "Meeting tomorrow"}],
            )

            entries = memory.get_all_conversations()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["text"], "Meeting tomorrow at 10 AM")

            reloaded = ConversationMemory(file_path=str(memory_path))
            self.assertEqual(len(reloaded.get_all_conversations()), 1)

    @patch.object(ConversationDatabase, "connect")
    def test_database_save_and_retrieval(self, mock_connect):
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (123,)

        database = ConversationDatabase()

        result = database.save_conversation(
            transcript="Meeting tomorrow at 10 AM",
            language="English",
            speaker="Speaker 1",
            events=[{"type": "Meeting", "text": "Meeting tomorrow"}],
            sentiment="Neutral",
        )

        self.assertEqual(result, 123)
        mock_connect.assert_called_once()
        mock_cursor.execute.assert_called_once()

        query = mock_cursor.execute.call_args[0][0]
        self.assertIn("INSERT INTO public.conversations", query)

    def test_knowledge_graph_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            db_path = Path(folder) / "test_graph.db"

            conn = sqlite3.connect(db_path)
            try:
                conn.execute("""
                    CREATE TABLE conversations (
                        id INTEGER PRIMARY KEY,
                        transcript TEXT,
                        language TEXT,
                        speaker TEXT,
                        events TEXT
                    )
                """)
                conn.execute(
                    """
                    INSERT INTO conversations
                    (id, transcript, language, speaker, events)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        1,
                        "Meeting tomorrow at 10 AM",
                        "English",
                        "Speaker 1",
                        "[{'type': 'Meeting', 'text': 'Meeting tomorrow'}]",
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            graph = KnowledgeGraph(db_path=str(db_path))
            graph.build_graph()

            node_ids = [node["id"] for node in graph.nodes]
            node_id_set = set(node_ids)

            self.assertGreater(len(graph.nodes), 0)
            self.assertEqual(len(node_ids), len(node_id_set))

            for edge in graph.relationships:
                self.assertIn(edge["source"], node_id_set)
                self.assertIn(edge["target"], node_id_set)


if __name__ == "__main__":
    unittest.main()
