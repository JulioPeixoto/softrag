import os
import sqlite3
import tempfile
import pytest
from src.softrag import softrag
class DummyEmbedModel:
    def embed_query(self, text):
        return [0.1] * 1536  

class DummyChatModel:
    def invoke(self, prompt):
        return f"Resposta simulada para: {prompt}"