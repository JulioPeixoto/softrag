"""
Basic tests for the softrag library.

This file contains unit tests to verify the core functionality
of softrag.
"""

import os
import unittest
import tempfile
from unittest.mock import MagicMock, patch

from softrag import Rag


class MockEmbed:
    """Mock for embedding model."""
    
    def embed_query(self, text):
        """Returns a fake embedding (unit vector)."""
        return [0.1] * 1536


class MockChat:
    """Mock for chat model."""
    
    def invoke(self, prompt):
        """Returns a fixed response."""
        return "This is a simulated response."


class TestSoftrag(unittest.TestCase):
    """Tests for the softrag library."""
    
    def setUp(self):
        """Setup for each test."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        
        self.embed_model = MockEmbed()
        self.chat_model = MockChat()
        self.rag = Rag(
            embed_model=self.embed_model,
            chat_model=self.chat_model,
            db_path=self.temp_db.name
        )
    
    def tearDown(self):
        """Cleanup after each test."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
        
        # Remove WAL files if they exist
        wal_file = f"{self.temp_db.name}-shm"
        if os.path.exists(wal_file):
            os.unlink(wal_file)
        
        wal_file = f"{self.temp_db.name}-wal"
        if os.path.exists(wal_file):
            os.unlink(wal_file)
    
    @patch("softrag.softrag.trafilatura.fetch_url")
    @patch("softrag.softrag.trafilatura.extract")
    def test_add_web(self, mock_extract, mock_fetch):
        """Tests adding web content."""
        mock_fetch.return_value = "<html>test</html>"
        mock_extract.return_value = "Extracted test content"
        
        self.rag.add_web("https://example.com")
        
        # Verify that the appropriate functions were called
        mock_fetch.assert_called_once_with("https://example.com")
        mock_extract.assert_called_once()
        
        # Query to see if content was added
        response = self.rag.query("test")
        self.assertEqual(response, "This is a simulated response.")
    
    def test_add_file_txt(self):
        """Tests adding a text file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("This is a test file for softrag.")
            temp_path = f.name
        
        try:
            self.rag.add_file(temp_path)
            response = self.rag.query("test")
            self.assertEqual(response, "This is a simulated response.")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main() 