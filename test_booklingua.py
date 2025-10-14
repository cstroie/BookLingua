#!/usr/bin/env python
# Unit tests for BookLingua - EPUB Book Translation Tool
# Copyright (C) 2025 Costin Stroie <costinstroie@eridu.eu.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import unittest
import tempfile
import os
import sqlite3
from unittest.mock import Mock, patch, MagicMock
from bs4 import BeautifulSoup

# Import the BookTranslator class
from booklingua import BookTranslator


class TestBookTranslator(unittest.TestCase):
    """Unit tests for the BookTranslator class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary database file
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        # Initialize the translator with a mock book path
        self.translator = BookTranslator(
            api_key="test-key",
            base_url="http://test-api.com/v1",
            model="test-model",
            verbose=False,
            book_path=self.temp_db.name
        )

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Close database connection
        if self.translator.conn:
            self.translator.conn.close()
        
        # Remove temporary database file
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

    def test_initialization(self):
        """Test BookTranslator initialization."""
        self.assertEqual(self.translator.api_key, "test-key")
        self.assertEqual(self.translator.base_url, "http://test-api.com/v1")
        self.assertEqual(self.translator.model, "test-model")
        self.assertFalse(self.translator.verbose)
        self.assertIsNotNone(self.translator.conn)
        self.assertEqual(self.translator.context, [])

    def test_get_language_code(self):
        """Test language code extraction."""
        self.assertEqual(self.translator.get_language_code("English"), "en")
        self.assertEqual(self.translator.get_language_code("French"), "fr")
        self.assertEqual(self.translator.get_language_code("German"), "ge")
        self.assertEqual(self.translator.get_language_code(""), "en")
        self.assertEqual(self.translator.get_language_code(None), "en")

    def test_strip_markdown_formatting(self):
        """Test markdown formatting removal."""
        # Test basic formatting removal
        text, prefix, suffix = self.translator.strip_markdown_formatting("**Hello**")
        self.assertEqual(text, "Hello")
        self.assertEqual(prefix, "**")
        self.assertEqual(suffix, "**")
        
        # Test with numbers and symbols
        text, prefix, suffix = self.translator.strip_markdown_formatting("1. Hello World!")
        self.assertEqual(text, "Hello World")
        self.assertEqual(prefix, "1. ")
        self.assertEqual(suffix, "!")
        
        # Test with no formatting
        text, prefix, suffix = self.translator.strip_markdown_formatting("Hello World")
        self.assertEqual(text, "Hello World")
        self.assertEqual(prefix, "")
        self.assertEqual(suffix, "")

    def test_remove_xml_tags(self):
        """Test XML tag removal."""
        # Test basic tag removal
        text = "<p>Hello <script>alert('test')</script> world</p>"
        cleaned = self.translator.remove_xml_tags(text, "script")
        self.assertEqual(cleaned, "<p>Hello  world</p>")
        
        # Test with self-closing tags
        text = "<p>Hello <br/> world</p>"
        cleaned = self.translator.remove_xml_tags(text, "br")
        self.assertEqual(cleaned, "<p>Hello  world</p>")
        
        # Test with no matching tags
        text = "<p>Hello world</p>"
        cleaned = self.translator.remove_xml_tags(text, "script")
        self.assertEqual(cleaned, "<p>Hello world</p>")

    def test_parse_chapter_numbers(self):
        """Test chapter number parsing."""
        available_chapters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # Test single chapter
        result = self.translator.parse_chapter_numbers("3", available_chapters)
        self.assertEqual(result, [3])
        
        # Test comma-separated chapters
        result = self.translator.parse_chapter_numbers("1,3,5", available_chapters)
        self.assertEqual(result, [1, 3, 5])
        
        # Test range
        result = self.translator.parse_chapter_numbers("3-7", available_chapters)
        self.assertEqual(result, [3, 4, 5, 6, 7])
        
        # Test mixed format
        result = self.translator.parse_chapter_numbers("1,3-5,8-10", available_chapters)
        self.assertEqual(result, [1, 3, 4, 5, 8, 9, 10])
        
        # Test with non-existent chapters
        result = self.translator.parse_chapter_numbers("1,15,20", available_chapters)
        self.assertEqual(result, [1])

    def test_context_management(self):
        """Test context management methods."""
        # Test context_add
        self.translator.context_add("Hello", "Bonjour")
        self.assertEqual(len(self.translator.context), 1)
        self.assertEqual(self.translator.context[0], ("Hello", "Bonjour"))
        
        # Test context_reset
        self.translator.context_reset()
        self.assertEqual(self.translator.context, [])
        
        # Test context_add with markdown cleaning
        self.translator.context_add("**Hello**", "**Bonjour**", clean=True)
        self.assertEqual(self.translator.context[0], ("Hello", "Bonjour"))

    def test_html_to_markdown_conversion(self):
        """Test HTML to Markdown conversion."""
        # Test basic HTML to Markdown conversion
        html = "<h1>Title</h1><p>This is a <strong>bold</strong> paragraph.</p>"
        soup = BeautifulSoup(html, 'html.parser')
        markdown = self.translator.html_to_markdown(soup)
        self.assertIn("# Title", markdown)
        self.assertIn("This is a **bold** paragraph.", markdown)

    def test_markdown_to_html_conversion(self):
        """Test Markdown to HTML conversion."""
        # Test basic Markdown to HTML conversion
        markdown = "# Title\n\nThis is a **bold** paragraph."
        title, html = self.translator.markdown_to_html(markdown)
        self.assertEqual(title, "Title")
        self.assertIn("<h1>Title</h1>", html)
        self.assertIn("<p>This is a <strong>bold</strong> paragraph.</p>", html)

    def test_calculate_fluency_score(self):
        """Test fluency score calculation."""
        # Test with good text
        good_text = "This is a well written sentence. This is another one."
        score = self.translator.calculate_fluency_score(good_text)
        self.assertIsInstance(score, int)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        
        # Test with empty text
        empty_text = ""
        score = self.translator.calculate_fluency_score(empty_text)
        self.assertEqual(score, 100)

    @patch('requests.post')
    def test_translate_text_with_cache(self, mock_post):
        """Test text translation with cache."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Bonjour le monde"}}]
        }
        mock_post.return_value = mock_response
        
        # Test translation without cache
        result = self.translator.translate_text("Hello world", "English", "French", use_cache=False)
        self.assertEqual(result[0], "Bonjour le monde")
        
        # Test translation with cache (should use cached result)
        with patch.object(self.translator, 'db_get_translation', return_value=("Hola mundo", 100, 95, "test-model")):
            result = self.translator.translate_text("Hello world", "English", "Spanish", use_cache=True)
            self.assertEqual(result[0], "Hola mundo")

    def test_database_operations(self):
        """Test database operations."""
        # Test database initialization
        self.assertIsNotNone(self.translator.conn)
        
        # Test saving and retrieving translations
        self.translator.db_save_translation(
            "Hello", "Bonjour", "English", "French", 1, 1, 1, 1000, 95, "test-model"
        )
        
        result = self.translator.db_get_translation("Hello", "English", "French")
        self.assertEqual(result[0], "Bonjour")
        self.assertEqual(result[1], 1000)
        self.assertEqual(result[2], 95)
        self.assertEqual(result[3], "test-model")

    def test_chapter_statistics(self):
        """Test chapter statistics methods."""
        # Add some test data
        self.translator.db_save_translation(
            "Hello", "Bonjour", "English", "French", 1, 1, 1, 1000, 95, "test-model"
        )
        self.translator.db_save_translation(
            "World", "Monde", "English", "French", 1, 1, 2, 1500, 90, "test-model"
        )
        
        # Test counting total paragraphs
        total = self.translator.db_count_total(1, 1, "English", "French")
        self.assertEqual(total, 2)
        
        # Test counting untranslated paragraphs
        untranslated = self.translator.db_count_untranslated(1, 1, "English", "French")
        self.assertEqual(untranslated, 0)  # Both are translated
        
        # Test chapter stats
        avg_time, elapsed_time, remaining_time = self.translator.db_chapter_stats(1, 1, "English", "French")
        self.assertEqual(avg_time, 1250.0)  # Average of 1000 and 1500
        self.assertEqual(elapsed_time, 2500.0)  # Sum of 1000 and 1500

    def test_set_console_width(self):
        """Test console width setting."""
        original_width = self.translator.console_width
        self.translator.set_console_width(100)
        self.assertEqual(self.translator.console_width, 100)
        self.assertEqual(len(self.translator.sep1), 100)
        self.assertEqual(len(self.translator.sep2), 100)
        self.assertEqual(len(self.translator.sep3), 100)
        
        # Test minimum width
        self.translator.set_console_width(10)
        self.assertEqual(self.translator.console_width, 20)  # Minimum is 20

if __name__ == '__main__':
    unittest.main()
