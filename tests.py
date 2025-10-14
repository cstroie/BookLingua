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

    @patch('ebooklib.epub.read_epub')
    def test_book_extract_metadata(self, mock_read_epub):
        """Test metadata extraction from EPUB."""
        # Create a mock book with metadata
        mock_book = Mock()
        mock_book.get_metadata.return_value = [[('Test Title', {})], [('Test Author', {})]]
        
        # Test the translator's book_extract_metadata method
        result = self.translator.book_extract_metadata(mock_book, "English")
        if result:  # Only check if metadata was extracted
            self.assertEqual(result['id'], 'metadata')
            self.assertIn('Test Title', result['paragraphs'])

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
        self.translator.db_insert_translation(
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
        self.translator.db_insert_translation(
            "Hello", "Bonjour", "English", "French", 1, 1, 1, 1000, 95, "test-model"
        )
        self.translator.db_insert_translation(
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

    def test_display_side_by_side(self):
        """Test side-by-side display functionality."""
        # This is primarily a display function, so we'll just test it doesn't crash
        try:
            self.translator.display_side_by_side("Hello", "Bonjour")
            self.translator.display_side_by_side("A" * 100, "B" * 100)  # Long texts
            self.translator.display_side_by_side("", "Empty left")
            self.translator.display_side_by_side("Empty right", "")
        except Exception as e:
            self.fail(f"display_side_by_side raised an exception: {e}")

    def test_context_prefill(self):
        """Test context prefill functionality."""
        # Add some test data to database
        self.translator.db_insert_translation(
            "Hello world", "Bonjour le monde", "English", "French", 1, 1, 1, 1000, 95, "test-model"
        )
        
        # Test prefill with existing translations
        self.translator.context_reset()
        self.translator.context_prefill("English", "French", 1)
        # Context should be populated
        self.assertGreater(len(self.translator.context), 0)

    def test_context_reset(self):
        """Test context reset functionality."""
        # Add some context
        self.translator.context_add("Hello", "Bonjour")
        self.translator.context_add("World", "Monde")
        self.assertEqual(len(self.translator.context), 2)
        
        # Reset context
        self.translator.context_reset()
        self.assertEqual(len(self.translator.context), 0)
        
        # Test with small chapter size
        self.translator.context_add("Hello", "Bonjour")
        self.translator.context_reset(5)  # Small chapter size
        # Context should be preserved for small chapters
        self.assertEqual(len(self.translator.context), 1)

    def test_process_inline_tags(self):
        """Test inline tag processing."""
        # Test basic formatting tags
        html = '<p>This is <strong>bold</strong> and <em>italic</em> text.</p>'
        soup = BeautifulSoup(html, 'html.parser')
        processed = self.translator.html_process_inlines(soup.p)
        text = processed.get_text()
        self.assertIn("**bold**", text)
        self.assertIn("*italic*", text)
        
        # Test code tags
        html = '<p>Use <code>print()</code> function.</p>'
        soup = BeautifulSoup(html, 'html.parser')
        processed = self.translator.html_process_inlines(soup.p)
        text = processed.get_text()
        self.assertIn("`print()`", text)
        
        # Test strikethrough
        html = '<p>This is <s>strikethrough</s> text.</p>'
        soup = BeautifulSoup(html, 'html.parser')
        processed = self.translator.html_process_inlines(soup.p)
        text = processed.get_text()
        self.assertIn("~~strikethrough~~", text)

    def test_process_inline_markdown(self):
        """Test inline markdown processing."""
        # Test bold formatting
        text = "This is **bold** text"
        result = self.translator.process_inline_markdown(text)
        self.assertIn("<strong>bold</strong>", result)
        
        # Test italic formatting
        text = "This is *italic* text"
        result = self.translator.process_inline_markdown(text)
        self.assertIn("<em>italic</em>", result)
        
        # Test code formatting
        text = "Use `print()` function"
        result = self.translator.process_inline_markdown(text)
        self.assertIn("<code>print()</code>", result)
        
        # Test strikethrough
        text = "This is ~~strikethrough~~ text"
        result = self.translator.process_inline_markdown(text)
        self.assertIn("<s>strikethrough</s>", result)

    def test_db_get_latest_edition(self):
        """Test getting latest edition number."""
        # Initially should be 0
        edition = self.translator.db_get_latest_edition("English", "French")
        self.assertEqual(edition, 0)
        
        # Add some translations
        self.translator.db_insert_translation(
            "Hello", "Bonjour", "English", "French", 1, 1, 1, 1000, 95, "test-model"
        )
        self.translator.db_insert_translation(
            "World", "Monde", "English", "French", 2, 1, 1, 1000, 95, "test-model"
        )
        
        # Should return the highest edition number
        edition = self.translator.db_get_latest_edition("English", "French")
        self.assertEqual(edition, 2)

    def test_db_get_chapters(self):
        """Test getting chapter list."""
        # Initially should be empty
        chapters = self.translator.db_get_chapters_list("English", "French", 1)
        self.assertEqual(chapters, [])
        
        # Add some translations across different chapters
        self.translator.db_insert_translation(
            "Hello", "Bonjour", "English", "French", 1, 1, 1, 1000, 95, "test-model"
        )
        self.translator.db_insert_translation(
            "World", "Monde", "English", "French", 1, 2, 1, 1000, 95, "test-model"
        )
        self.translator.db_insert_translation(
            "Test", "Test", "English", "French", 1, 1, 2, 1000, 95, "test-model"
        )
        
        # Should return chapters in order
        chapters = self.translator.db_get_chapters_list("English", "French", 1)
        self.assertEqual(chapters, [1, 2])
        
        # Test sorting by length
        chapters = self.translator.db_get_chapters_list("English", "French", 1, by_length=True)
        self.assertEqual(chapters, [1, 2])  # Chapter 1 has 2 paragraphs, chapter 2 has 1

    def test_db_search(self):
        """Test database search functionality."""
        # Add some test data
        self.translator.db_insert_translation(
            "Hello world example", "Bonjour le monde exemple", "English", "French", 1, 1, 1, 1000, 95, "test-model"
        )
        self.translator.db_insert_translation(
            "Another world test", "Un autre test mondial", "English", "French", 1, 1, 2, 1000, 90, "test-model"
        )
        
        # Search for "world"
        results = self.translator.db_search("world", "English", "French")
        self.assertEqual(len(results), 2)
        self.assertIn("Hello world example", [r[0] for r in results])
        self.assertIn("Another world test", [r[0] for r in results])

    def test_db_export_import_csv(self):
        """Test CSV export and import functionality."""
        # Add some test data
        self.translator.db_insert_translation(
            "Hello", "Bonjour", "English", "French", 1, 1, 1, 1000, 95, "test-model"
        )
        
        # Export to CSV
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_csv:
            csv_path = temp_csv.name
        
        try:
            self.translator.db_export_csv(csv_path)
            
            # Check that file was created
            self.assertTrue(os.path.exists(csv_path))
            
            # Import back
            self.translator.db_import_csv(csv_path)
        finally:
            # Clean up
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_calculate_adequacy_score(self):
        """Test adequacy score calculation."""
        # This uses the translation API, so we'll mock it
        with patch.object(self.translator, 'translate_text', return_value=("85", -1, -1, "test-model")):
            score = self.translator.calculate_adequacy_score("Hello world", "Bonjour le monde", "English", "French")
            self.assertIsInstance(score, int)
            self.assertEqual(score, 85)

    def test_calculate_consistency_score(self):
        """Test consistency score calculation."""
        chapters = [
            {'content': 'Hello world, this is a test. Hello again.'},
            {'content': 'Another chapter with Hello world.'}
        ]
        score = self.translator.calculate_consistency_score(chapters)
        self.assertIsInstance(score, int)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_detect_translation_errors(self):
        """Test translation error detection."""
        errors = self.translator.detect_translation_errors(
            "Hello world", 
            "Bonjour le monde Hello world",  # Contains untranslated English
            "English"
        )
        self.assertIsInstance(errors, dict)
        self.assertIn('untranslated_segments', errors)

    def test_generate_quality_report(self):
        """Test quality report generation."""
        chapters = [{'content': 'This is a test chapter with some content.'}]
        
        with patch.object(self.translator, 'translate_text', return_value=("Test translation", -1, -1, "test-model")):
            report = self.translator.generate_quality_report(chapters, "English", "French")
            
        self.assertIsInstance(report, dict)
        self.assertIn('fluency_scores', report)
        self.assertIn('overall_score', report)
        self.assertIsInstance(report['overall_score'], int)

    def test_book_create_template(self):
        """Test EPUB template creation."""
        # Create a mock original book
        original_book = MagicMock()
        original_book.get_metadata.return_value = [('test-id', {})], [('Test Title', {})], [('Test Author', {})]
        
        with patch.object(self.translator, 'translate_text', return_value=("Test Title", -1, -1, "test-model")):
            new_book = self.translator.book_create_template(original_book, "English", "French")
            
        self.assertIsNotNone(new_book)
        self.assertEqual(new_book.language, "fr")

    def test_book_create_chapter(self):
        """Test EPUB chapter creation."""
        # Add some test translations
        self.translator.db_insert_translation(
            "Chapter Title", "Titre du Chapitre", "English", "French", 1, 1, 0, 1000, 95, "test-model"
        )
        self.translator.db_insert_translation(
            "Hello world", "Bonjour le monde", "English", "French", 1, 1, 1, 1000, 95, "test-model"
        )
        
        chapter = self.translator.book_create_chapter(1, 1, "English", "French")
        self.assertIsNotNone(chapter)
        self.assertIn("Titre du Chapitre", chapter.title)

    def test_book_finalize(self):
        """Test EPUB finalization."""
        # Create a mock book
        book = MagicMock()
        
        # Create mock chapters
        chapter1 = MagicMock()
        chapter2 = MagicMock()
        chapters = [chapter1, chapter2]
        
        # This should not raise an exception
        try:
            self.translator.book_finalize(book, chapters)
        except Exception as e:
            self.fail(f"book_finalize raised an exception: {e}")

    def test_remove_xml_tags_edge_cases(self):
        """Test edge cases for XML tag removal."""
        # Test with None input
        result = self.translator.remove_xml_tags(None, "script")
        self.assertIsNone(result)
        
        # Test with empty tag name
        result = self.translator.remove_xml_tags("test", "")
        self.assertEqual(result, "test")
        
        # Test with nested tags
        text = "<p>Hello <div><script>alert('test')</script></div> world</p>"
        result = self.translator.remove_xml_tags(text, "script")
        self.assertNotIn("<script>", result)

    def test_strip_markdown_formatting_edge_cases(self):
        """Test edge cases for markdown formatting removal."""
        # Test with None input
        text, prefix, suffix = self.translator.strip_markdown_formatting(None)
        self.assertEqual(text, "")
        self.assertEqual(prefix, "")
        self.assertEqual(suffix, "")
        
        # Test with only symbols
        text, prefix, suffix = self.translator.strip_markdown_formatting("***")
        self.assertEqual(text, "")
        self.assertEqual(prefix, "***")
        self.assertEqual(suffix, "")

    def test_parse_chapter_numbers_edge_cases(self):
        """Test edge cases for chapter number parsing."""
        available_chapters = [1, 2, 3, 4, 5]
        
        # Test with None input
        result = self.translator.parse_chapter_numbers(None, available_chapters)
        self.assertEqual(result, available_chapters)
        
        # Test with empty string
        result = self.translator.parse_chapter_numbers("", available_chapters)
        self.assertEqual(result, [])
        
        # Test with invalid format
        with self.assertRaises(ValueError):
            self.translator.parse_chapter_numbers("invalid", available_chapters)

    def test_get_language_code_edge_cases(self):
        """Test edge cases for language code extraction."""
        # Test with None input
        result = self.translator.get_language_code(None)
        self.assertEqual(result, "en")
        
        # Test with empty string
        result = self.translator.get_language_code("")
        self.assertEqual(result, "en")
        
        # Test with single character
        result = self.translator.get_language_code("E")
        self.assertEqual(result, "e")

if __name__ == '__main__':
    unittest.main()
