#!/usr/bin/env python
#import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from openai import OpenAI
import os
from typing import List, Dict
from datetime import datetime

class EPUBTranslator:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "gpt-4o"):
        """
        Initialize with OpenAI-compatible API
        
        Args:
            api_key: Your API key
            base_url: Base URL for the API (e.g., "https://api.openai.com/v1" for OpenAI,
                     "http://localhost:11434/v1" for Ollama, etc.)
            model: Model name to use (e.g., "gpt-4o", "qwen2.5:72b", "mistral-large-latest")
        """
        self.client = OpenAI(
            api_key=api_key or os.environ.get('OPENAI_API_KEY', 'dummy-key'),
            base_url=base_url
        )
        self.model = model
        print(f"Initialized with model: {model}")
        if base_url:
            print(f"Using API endpoint: {base_url}")
    
    def extract_text_from_epub(self, epub_path: str) -> List[dict]:
        """Extract text content from EPUB file"""
        book = epub.read_epub(epub_path)
        chapters = []
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text(separator='\n', strip=True)
                
                if text:  # Only include non-empty chapters
                    chapters.append({
                        'id': item.get_id(),
                        'name': item.get_name(),
                        'content': text,
                        'html': item.get_content()
                    })
        
        return chapters
    
    def translate_text(self, text: str, target_lang: str, source_lang: str = "English", 
                       chunk_size: int = 3000) -> str:
        """Translate text in chunks using OpenAI-compatible API"""
        if len(text) <= chunk_size:
            return self._translate_chunk(text, target_lang, source_lang)
        
        # Split into chunks at paragraph boundaries
        paragraphs = text.split('\n\n')
        translated_parts = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            if current_length + para_length > chunk_size and current_chunk:
                # Translate current chunk
                chunk_text = '\n\n'.join(current_chunk)
                translated_parts.append(self._translate_chunk(chunk_text, target_lang, source_lang))
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length
        
        # Translate remaining chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            translated_parts.append(self._translate_chunk(chunk_text, target_lang, source_lang))
        
        return '\n\n'.join(translated_parts)
    
    def _translate_chunk(self, text: str, target_lang: str, source_lang: str) -> str:
        """Translate a single chunk of text"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": f"""Translate the following {source_lang} text to {target_lang}. 
Maintain the original formatting, paragraph breaks, and tone. 
Only provide the translation, no explanations.

Text to translate:
{text}"""
                }],
                temperature=0.3,  # Lower temperature for more consistent translations
                max_tokens=8000
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error during translation: {e}")
            raise
    
    def translate_direct(self, text: str) -> str:
        """Direct English to Romanian translation"""
        return self.translate_text(text, "Romanian", "English")
    
    def translate_pivot(self, text: str) -> Dict[str, str]:
        """Pivot translation: English -> French -> Romanian"""
        print("  - Translating to French...")
        french = self.translate_text(text, "French", "English")
        print("  - Translating French to Romanian...")
        romanian = self.translate_text(french, "Romanian", "French")
        return {
            'french': french,
            'romanian': romanian
        }
    
    def create_comparison_html(self, chapter_num: int, original: str, 
                              direct: str, pivot_french: str, pivot_romanian: str) -> str:
        """Create HTML comparison of translations"""
        html = f"""
        <div class="chapter-comparison">
            <h2>Chapter {chapter_num} Comparison</h2>
            
            <div class="translation-block">
                <h3>Original English</h3>
                <div class="text-content">{original.replace(chr(10), '<br>')}</div>
            </div>
            
            <div class="translation-block">
                <h3>Direct Translation (English → Romanian)</h3>
                <div class="text-content direct">{direct.replace(chr(10), '<br>')}</div>
            </div>
            
            <div class="translation-block">
                <h3>Intermediate French (English → French)</h3>
                <div class="text-content french">{pivot_french.replace(chr(10), '<br>')}</div>
            </div>
            
            <div class="translation-block">
                <h3>Pivot Translation (English → French → Romanian)</h3>
                <div class="text-content pivot">{pivot_romanian.replace(chr(10), '<br>')}</div>
            </div>
        </div>
        <hr>
        """
        return html
    
    def translate_epub_with_comparison(self, input_path: str, output_dir: str = "output", 
                                       mode: str = "both"):
        """
        Translate EPUB using direct, pivot, or both methods
        
        Args:
            input_path: Path to input EPUB file
            output_dir: Directory for output files
            mode: "direct", "pivot", or "both" (default: "both")
        """
        if mode not in ["direct", "pivot", "both"]:
            raise ValueError("mode must be 'direct', 'pivot', or 'both'")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Reading EPUB from {input_path}...")
        book = epub.read_epub(input_path)
        chapters = self.extract_text_from_epub(input_path)
        
        print(f"Found {len(chapters)} chapters to translate")
        print(f"Translation mode: {mode.upper()}\n")
        
        # Prepare output books based on mode
        direct_book = None
        pivot_book = None
        
        if mode in ["direct", "both"]:
            direct_book = self._create_book_template(book, "Direct Translation")
        if mode in ["pivot", "both"]:
            pivot_book = self._create_book_template(book, "Pivot Translation")
        
        # HTML comparison document (only for "both" mode)
        comparison_html = None
        if mode == "both":
            comparison_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Translation Comparison</title>
                <style>
                    body { font-family: Georgia, serif; line-height: 1.6; max-width: 1400px; margin: 0 auto; padding: 20px; }
                    h2 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                    h3 { color: #34495e; margin-top: 20px; }
                    .chapter-comparison { margin-bottom: 50px; }
                    .translation-block { margin: 20px 0; padding: 15px; border-radius: 5px; }
                    .translation-block h3 { margin-top: 0; }
                    .text-content { background: #f8f9fa; padding: 15px; border-left: 4px solid #ccc; }
                    .direct { border-left-color: #27ae60; }
                    .french { border-left-color: #3498db; }
                    .pivot { border-left-color: #e74c3c; }
                    hr { margin: 40px 0; border: none; border-top: 2px dashed #ccc; }
                </style>
            </head>
            <body>
                <h1>English to Romanian Translation Comparison</h1>
                <p><strong>Generated:</strong> """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                <p><strong>Model:</strong> """ + self.model + """</p>
                <p><strong>Method Comparison:</strong></p>
                <ul>
                    <li><span style="color: #27ae60;">●</span> <strong>Direct:</strong> English → Romanian (single step)</li>
                    <li><span style="color: #e74c3c;">●</span> <strong>Pivot:</strong> English → French → Romanian (two steps)</li>
                </ul>
            """
        
        direct_chapters = []
        pivot_chapters = []
        
        # Process each chapter
        for i, chapter in enumerate(chapters):
            print(f"\n{'='*60}")
            print(f"Chapter {i+1}/{len(chapters)}: {chapter['name']}")
            print(f"{'='*60}")
            
            original_text = chapter['content']
            
            direct_translation = None
            pivot_result = None
            
            # Direct translation
            if mode in ["direct", "both"]:
                print("Direct translation (English → Romanian)...")
                direct_translation = self.translate_direct(original_text)
            
            # Pivot translation
            if mode in ["pivot", "both"]:
                print("Pivot translation (English → French → Romanian)...")
                pivot_result = self.translate_pivot(original_text)
            
            # Add to comparison HTML (only in "both" mode)
            if mode == "both":
                comparison_html += self.create_comparison_html(
                    i + 1, 
                    original_text, 
                    direct_translation,
                    pivot_result['french'],
                    pivot_result['romanian']
                )
            
            # Create chapters for books
            if direct_book and direct_translation:
                direct_chapter = epub.EpubHtml(
                    title=f'Chapter {i+1}',
                    file_name=f'chapter_{i+1}.xhtml',
                    lang='ro'
                )
                direct_chapter.content = f'<html><body>{self._text_to_html(direct_translation)}</body></html>'
                direct_book.add_item(direct_chapter)
                direct_chapters.append(direct_chapter)
            
            if pivot_book and pivot_result:
                pivot_chapter = epub.EpubHtml(
                    title=f'Chapter {i+1}',
                    file_name=f'chapter_{i+1}.xhtml',
                    lang='ro'
                )
                pivot_chapter.content = f'<html><body>{self._text_to_html(pivot_result["romanian"])}</body></html>'
                pivot_book.add_item(pivot_chapter)
                pivot_chapters.append(pivot_chapter)
            
            print(f"✓ Chapter {i+1} complete")
        
        # Finalize books
        if direct_book:
            self._finalize_book(direct_book, direct_chapters)
        if pivot_book:
            self._finalize_book(pivot_book, pivot_chapters)
        
        # Save outputs
        print(f"\n{'='*60}")
        print("Saving output files...")
        
        if direct_book:
            direct_path = os.path.join(output_dir, "direct_translation.epub")
            epub.write_epub(direct_path, direct_book)
            print(f"✓ Direct translation saved: {direct_path}")
        
        if pivot_book:
            pivot_path = os.path.join(output_dir, "pivot_translation.epub")
            epub.write_epub(pivot_path, pivot_book)
            print(f"✓ Pivot translation saved: {pivot_path}")
        
        if mode == "both":
            comparison_path = os.path.join(output_dir, "comparison.html")
            comparison_html += "</body></html>"
            with open(comparison_path, 'w', encoding='utf-8') as f:
                f.write(comparison_html)
            print(f"✓ Comparison document saved: {comparison_path}")
        
        print(f"\n{'='*60}")
        print("Translation complete! 🎉")
        print(f"{'='*60}")
    
    def _create_book_template(self, original_book, method_name: str):
        """Create a new EPUB book with metadata from original"""
        new_book = epub.EpubBook()
        new_book.set_identifier(original_book.get_metadata('DC', 'identifier')[0][0])
        
        original_title = original_book.get_metadata('DC', 'title')[0][0]
        new_book.set_title(f"{original_title} (Română - {method_name})")
        new_book.set_language('ro')
        
        for author in original_book.get_metadata('DC', 'creator'):
            new_book.add_author(author[0])
        
        return new_book
    
    def _finalize_book(self, book, chapters):
        """Add navigation and finalize book structure"""
        book.toc = tuple(chapters)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = ['nav'] + chapters
    
    def _text_to_html(self, text: str) -> str:
        """Convert plain text to HTML paragraphs"""
        paragraphs = text.split('\n\n')
        html_paragraphs = [f'<p>{p.replace(chr(10), "<br/>")}</p>' for p in paragraphs if p.strip()]
        return '\n'.join(html_paragraphs)

# Usage examples
if __name__ == "__main__":
    
    # Example 1: Using OpenAI
    translator_openai = EPUBTranslator(
        api_key=os.environ.get('OPENAI_API_KEY'),
        base_url="https://api.openai.com/v1",
        model="gpt-4o"
    )
    
    # Example 2: Using Ollama (local)
    # translator_ollama = EPUBTranslator(
    #     base_url="http://localhost:11434/v1",
    #     model="qwen2.5:72b"
    # )
    
    # Example 3: Using Mistral AI
    # translator_mistral = EPUBTranslator(
    #     api_key=os.environ.get('MISTRAL_API_KEY'),
    #     base_url="https://api.mistral.ai/v1",
    #     model="mistral-large-latest"
    # )
    
    # Example 4: Using DeepSeek
    # translator_deepseek = EPUBTranslator(
    #     api_key=os.environ.get('DEEPSEEK_API_KEY'),
    #     base_url="https://api.deepseek.com/v1",
    #     model="deepseek-chat"
    # )
    
    # Example 5: Using LM Studio (local)
    # translator_lmstudio = EPUBTranslator(
    #     base_url="http://localhost:1234/v1",
    #     model="qwen2.5-72b"
    # )
    
    # Example 6: Using Together AI
    # translator_together = EPUBTranslator(
    #     api_key=os.environ.get('TOGETHER_API_KEY'),
    #     base_url="https://api.together.xyz/v1",
    #     model="Qwen/Qwen2.5-72B-Instruct-Turbo"
    # )
    
    # Run the translation with your preferred mode
    
    # Option 1: Direct translation only (faster, cheaper)
    # translator_openai.translate_epub_with_comparison('input.epub', output_dir='output', mode='direct')
    
    # Option 2: Pivot translation only (to test French intermediary)
    # translator_openai.translate_epub_with_comparison('input.epub', output_dir='output', mode='pivot')
    
    # Option 3: Both methods with comparison (most expensive but lets you compare)
    translator_openai.translate_epub_with_comparison('input.epub', output_dir='output', mode='both')
