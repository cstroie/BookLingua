import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import anthropic
import os
from typing import List

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
    
    def translate_text(self, text: str, chunk_size: int = 3000) -> str:
        """Translate text in chunks using Claude"""
        if len(text) <= chunk_size:
            return self._translate_chunk(text)
        
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
                translated_parts.append(self._translate_chunk(chunk_text))
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length
        
        # Translate remaining chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            translated_parts.append(self._translate_chunk(chunk_text))
        
        return '\n\n'.join(translated_parts)
    
    def _translate_chunk(self, text: str) -> str:
        """Translate a single chunk of text"""
        message = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8000,
            messages=[{
                "role": "user",
                "content": f"""Translate the following English text to Romanian. 
Maintain the original formatting, paragraph breaks, and tone. 
Only provide the translation, no explanations.

Text to translate:
{text}"""
            }]
        )
        
        return message.content[0].text
    
    def translate_epub(self, input_path: str, output_path: str):
        """Translate entire EPUB file"""
        print(f"Reading EPUB from {input_path}...")
        book = epub.read_epub(input_path)
        chapters = self.extract_text_from_epub(input_path)
        
        print(f"Found {len(chapters)} chapters to translate")
        
        # Create new book with same metadata
        new_book = epub.EpubBook()
        new_book.set_identifier(book.get_metadata('DC', 'identifier')[0][0])
        new_book.set_title(book.get_metadata('DC', 'title')[0][0] + ' (Română)')
        new_book.set_language('ro')
        
        # Copy authors
        for author in book.get_metadata('DC', 'creator'):
            new_book.add_author(author[0])
        
        # Translate and add chapters
        new_chapters = []
        for i, chapter in enumerate(chapters):
            print(f"Translating chapter {i+1}/{len(chapters)}: {chapter['name']}")
            
            translated_text = self.translate_text(chapter['content'])
            
            # Create new chapter
            new_chapter = epub.EpubHtml(
                title=f'Chapter {i+1}',
                file_name=chapter['name'],
                lang='ro'
            )
            new_chapter.content = f'<html><body><p>{translated_text.replace(chr(10), "</p><p>")}</p></body></html>'
            new_book.add_item(new_chapter)
            new_chapters.append(new_chapter)
        
        # Add navigation
        new_book.toc = tuple(new_chapters)
        new_book.add_item(epub.EpubNcx())
        new_book.add_item(epub.EpubNav())
        
        # Set spine
        new_book.spine = ['nav'] + new_chapters
        
        # Write the translated EPUB
        print(f"Writing translated EPUB to {output_path}...")
        epub.write_epub(output_path, new_book)
        print("Translation complete!")

# Usage example
if __name__ == "__main__":
    # Set your API key
    API_KEY = os.environ.get('ANTHROPIC_API_KEY', 'your-api-key-here')
    
    translator = EPUBTranslator(API_KEY)
    translator.translate_epub('input.epub', 'output_romanian.epub')
