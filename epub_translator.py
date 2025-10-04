import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import anthropic
import os
from typing import List, Dict
import json
from datetime import datetime

class EPUBTranslator:
    def __init__(self, api_key: str):
        """Initialize with your Anthropic API key"""
        self.client = anthropic.Anthropic(api_key=api_key)
    
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
        """Translate text in chunks using Claude"""
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
        message = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8000,
            messages=[{
                "role": "user",
                "content": f"""Translate the following {source_lang} text to {target_lang}. 
Maintain the original formatting, paragraph breaks, and tone. 
Only provide the translation, no explanations.

Text to translate:
{text}"""
            }]
        )
        
        return message.content[0].text
    
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
    
    def translate_epub_with_comparison(self, input_path: str, output_dir: str = "output"):
        """Translate EPUB using both methods and create comparison"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Reading EPUB from {input_path}...")
        book = epub.read_epub(input_path)
        chapters = self.extract_text_from_epub(input_path)
        
        print(f"Found {len(chapters)} chapters to translate\n")
        
        # Prepare output books
        direct_book = self._create_book_template(book, "Direct Translation")
        pivot_book = self._create_book_template(book, "Pivot Translation")
        
        # HTML comparison document
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
            
            # Direct translation
            print("Direct translation (English → Romanian)...")
            direct_translation = self.translate_direct(original_text)
            
            # Pivot translation
            print("Pivot translation (English → French → Romanian)...")
            pivot_result = self.translate_pivot(original_text)
            
            # Add to comparison HTML
            comparison_html += self.create_comparison_html(
                i + 1, 
                original_text, 
                direct_translation,
                pivot_result['french'],
                pivot_result['romanian']
            )
            
            # Create chapters for both books
            direct_chapter = epub.EpubHtml(
                title=f'Chapter {i+1}',
                file_name=f'chapter_{i+1}.xhtml',
                lang='ro'
            )
            direct_chapter.content = f'<html><body>{self._text_to_html(direct_translation)}</body></html>'
            direct_book.add_item(direct_chapter)
            direct_chapters.append(direct_chapter)
            
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
        self._finalize_book(direct_book, direct_chapters)
        self._finalize_book(pivot_book, pivot_chapters)
        
        # Save outputs
        direct_path = os.path.join(output_dir, "direct_translation.epub")
        pivot_path = os.path.join(output_dir, "pivot_translation.epub")
        comparison_path = os.path.join(output_dir, "comparison.html")
        
        print(f"\n{'='*60}")
        print("Saving output files...")
        epub.write_epub(direct_path, direct_book)
        print(f"✓ Direct translation saved: {direct_path}")
        
        epub.write_epub(pivot_path, pivot_book)
        print(f"✓ Pivot translation saved: {pivot_path}")
        
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

# Usage example
if __name__ == "__main__":
    # Set your API key
    API_KEY = os.environ.get('ANTHROPIC_API_KEY', 'your-api-key-here')
    
    translator = EPUBTranslator(API_KEY)
    
    # This will create:
    # - output/direct_translation.epub (English → Romanian)
    # - output/pivot_translation.epub (English → French → Romanian)
    # - output/comparison.html (side-by-side comparison)
    
    translator.translate_epub_with_comparison('input.epub', output_dir='output')
