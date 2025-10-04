#!/usr/bin/env python
# BookLingua - Translate EPUB books using AI models
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

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import requests
import json
import os
import argparse
import re
from typing import List, Dict
from datetime import datetime

class EPUBTranslator:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "gpt-4o", verbose: bool = False):
        """
        Initialize with OpenAI-compatible API
        
        Args:
            api_key: Your API key
            base_url: Base URL for the API (e.g., "https://api.openai.com/v1" for OpenAI,
                     "http://localhost:11434/v1" for Ollama, etc.)
            model: Model name to use (e.g., "gpt-4o", "qwen2.5:72b", "mistral-large-latest")
            verbose: Whether to print verbose output
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY', 'dummy-key')
        self.base_url = base_url or "https://api.openai.com/v1"
        self.model = model
        self.verbose = verbose
        print(f"Initialized with model: {model}")
        if base_url:
            print(f"Using API endpoint: {base_url}")
    
    def extract_text_from_epub(self, epub_path: str) -> List[dict]:
        """Extract text content from EPUB file"""
        book = epub.read_epub(epub_path)
        chapters = []
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                html_content = item.get_content()
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Convert HTML to Markdown
                markdown_content = self._html_to_markdown(soup)
                
                # Extract paragraphs from Markdown
                paragraphs = [p.strip() for p in markdown_content.split('\n\n') if p.strip()]
                
                if markdown_content.strip():  # Only include non-empty chapters
                    chapters.append({
                        'id': item.get_id(),
                        'name': item.get_name(),
                        'content': markdown_content,
                        'html': html_content,
                        'paragraphs': paragraphs
                    })
        
        return chapters
    
    def _html_to_markdown(self, soup) -> str:
        """Convert HTML to Markdown format"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        markdown_lines = []
        
        # Process each element
        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div', 'br']):
            # Process inline tags within the element
            processed_element = self._process_inline_tags(element)
            text = processed_element.get_text(separator=' ', strip=True)
            if not text:
                continue
                
            # Add appropriate Markdown formatting
            if element.name.startswith('h'):
                level = int(element.name[1])
                markdown_lines.append('#' * level + ' ' + text)
            elif element.name == 'li':
                markdown_lines.append('- ' + text)
            else:
                markdown_lines.append(text)
        
        # Join with double newlines for paragraph separation
        return '\n\n'.join(markdown_lines)
    
    def _process_inline_tags(self, element) -> BeautifulSoup:
        """Process inline HTML tags and convert them to Markdown-style formatting"""
        # Create a copy to avoid modifying the original
        element_copy = BeautifulSoup(str(element), 'html.parser')
        
        # Process each inline tag
        for tag in element_copy.find_all(['i', 'em', 'b', 'strong', 'u', 'ins', 's', 'del', 'code', 'span']):
            text = tag.get_text()
            if not text:
                continue
                
            # Replace with appropriate Markdown formatting
            if tag.name in ['i', 'em']:
                replacement = f'*{text}*'
            elif tag.name in ['b', 'strong']:
                replacement = f'**{text}**'
            elif tag.name in ['u', 'ins']:
                replacement = f'__{text}__'
            elif tag.name in ['s', 'del']:
                replacement = f'~~{text}~~'
            elif tag.name == 'code':
                replacement = f'`{text}`'
            elif tag.name == 'span':
                # Check for styling that mimics other tags
                style = tag.get('style', '').lower()
                css_class = tag.get('class', [])
                if isinstance(css_class, list):
                    css_class = ' '.join(css_class).lower()
                else:
                    css_class = css_class.lower()
                
                # Check for bold styling
                if ('font-weight' in style and 'bold' in style) or any(cls in css_class for cls in ['bold', 'strong']):
                    replacement = f'**{text}**'
                # Check for italic styling
                elif ('font-style' in style and 'italic' in style) or any(cls in css_class for cls in ['italic', 'em']):
                    replacement = f'*{text}*'
                # Check for underline styling
                elif ('text-decoration' in style and 'underline' in style) or any(cls in css_class for cls in ['underline']):
                    replacement = f'__{text}__'
                # Check for strikethrough styling
                elif ('text-decoration' in style and 'line-through' in style) or any(cls in css_class for cls in ['strikethrough', 'line-through']):
                    replacement = f'~~{text}~~'
                # Check for monospace styling
                elif ('font-family' in style and ('monospace' in style or 'courier' in style)) or any(cls in css_class for cls in ['code', 'monospace']):
                    replacement = f'`{text}`'
                else:
                    # Default to plain text for other spans
                    replacement = text
            else:  # other tags
                replacement = text
            
            # Replace the tag with formatted text
            tag.replace_with(replacement)
        
        return element_copy
    
    def _markdown_to_html(self, markdown_text: str) -> str:
        """Convert Markdown back to HTML"""
        lines = markdown_text.split('\n')
        html_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Handle headers
            if line.startswith('###### '):
                content = self._process_markdown_inline_formatting(line[7:])
                html_lines.append(f'<h6>{content}</h6>')
            elif line.startswith('##### '):
                content = self._process_markdown_inline_formatting(line[6:])
                html_lines.append(f'<h5>{content}</h5>')
            elif line.startswith('#### '):
                content = self._process_markdown_inline_formatting(line[5:])
                html_lines.append(f'<h4>{content}</h4>')
            elif line.startswith('### '):
                content = self._process_markdown_inline_formatting(line[4:])
                html_lines.append(f'<h3>{content}</h3>')
            elif line.startswith('## '):
                content = self._process_markdown_inline_formatting(line[3:])
                html_lines.append(f'<h2>{content}</h2>')
            elif line.startswith('# '):
                content = self._process_markdown_inline_formatting(line[2:])
                html_lines.append(f'<h1>{content}</h1>')
            # Handle lists
            elif line.startswith('- '):
                content = self._process_markdown_inline_formatting(line[2:])
                html_lines.append(f'<li>{content}</li>')
            # Handle regular paragraphs
            else:
                content = self._process_markdown_inline_formatting(line)
                html_lines.append(f'<p>{content}</p>')
        
        return '\n'.join(html_lines)
    
    def _process_markdown_inline_formatting(self, text: str) -> str:
        """Convert Markdown inline formatting back to HTML tags"""
        # Process formatting in order of precedence
        # Code (highest precedence)
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        # Strikethrough
        text = re.sub(r'~~([^~]+)~~', r'<s>\1</s>', text)
        # Strong/Bold
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        # Emphasis/Italic
        text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
        # Underline
        text = re.sub(r'__([^_]+)__', r'<u>\1</u>', text)
        
        return text
    
    def translate_text(self, text: str, target_lang: str, source_lang: str = "English", 
                       chunk_size: int = 3000) -> str:
        """Translate text in chunks using OpenAI-compatible API"""
        # Split into paragraphs (separated by double newlines)
        paragraphs = text.split('\n\n')
        
        # If we have only one paragraph or the total text is small enough, translate as one chunk
        if len(paragraphs) <= 1 or len(text) <= chunk_size:
            return self._translate_chunk(text, target_lang, source_lang)
        
        # Otherwise, process each paragraph as a separate chunk
        translated_paragraphs = []
        for i, paragraph in enumerate(paragraphs):
            if self.verbose:
                print(f"Translating paragraph {i+1}/{len(paragraphs)}")
            if paragraph.strip():  # Only translate non-empty paragraphs
                translated_paragraphs.append(self._translate_chunk(paragraph, target_lang, source_lang))
            else:
                # Preserve empty paragraphs (section breaks)
                translated_paragraphs.append(paragraph)
        
        return '\n\n'.join(translated_paragraphs)
    
    def _translate_chunk(self, text: str, target_lang: str, source_lang: str) -> str:
        """Translate a single chunk of text"""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            # Add API key if provided and not a local endpoint
            if self.api_key and self.api_key != 'dummy-key':
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            payload = {
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": f"""/no_think Translate the following {source_lang} text to {target_lang}. 
Maintain the original formatting, paragraph breaks, and tone. 
Only provide the translation, no explanations.

Text to translate:
{text}"""
                }],
                "temperature": 0.3,  # Lower temperature for more consistent translations
                "max_tokens": 8000
            }
            
            if self.verbose:
                print(f"\n--- Translating chunk ---")
                print(f"Source language: {source_lang}")
                print(f"Target language: {target_lang}")
                print(f"Text to translate:\n{text}\n")
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
            
            result = response.json()
            translation = result["choices"][0]["message"]["content"].strip()
            
            # Remove any text between <think> and </think> tags
            import re
            translation = re.sub(r'<think>.*?</think>', '', translation, flags=re.DOTALL).strip()
            
            if self.verbose:
                print(f"Translation:\n{translation}\n")
                print("--- End of chunk translation ---\n")
            
            return translation
        except Exception as e:
            print(f"Error during translation: {e}")
            raise
    
    def translate_direct(self, text: str, source_lang: str = "English", target_lang: str = "Romanian") -> str:
        """Direct translation from source to target language"""
        return self.translate_text(text, target_lang, source_lang)
    
    def translate_pivot(self, text: str, source_lang: str = "English", pivot_lang: str = "French", 
                       target_lang: str = "Romanian") -> Dict[str, str]:
        """Pivot translation: source -> pivot -> target"""
        print(f"  - Translating to {pivot_lang}...")
        intermediate = self.translate_text(text, pivot_lang, source_lang)
        print(f"  - Translating {pivot_lang} to {target_lang}...")
        final = self.translate_text(intermediate, target_lang, pivot_lang)
        return {
            'intermediate': intermediate,
            'final': final
        }
    
    def create_comparison_html(self, chapter_num: int, original: str, 
                              direct: str, pivot_intermediate: str, pivot_final: str,
                              source_lang: str = "English", pivot_lang: str = "French", 
                              target_lang: str = "Romanian") -> str:
        """Create HTML comparison of translations"""
        html = f"""
        <div class="chapter-comparison">
            <h2>Chapter {chapter_num} Comparison</h2>
            
            <div class="translation-block">
                <h3>Original {source_lang}</h3>
                <div class="text-content">{original.replace(chr(10), '<br>')}</div>
            </div>
            
            <div class="translation-block">
                <h3>Direct Translation ({source_lang} → {target_lang})</h3>
                <div class="text-content direct">{direct.replace(chr(10), '<br>')}</div>
            </div>
            
            <div class="translation-block">
                <h3>Intermediate {pivot_lang} ({source_lang} → {pivot_lang})</h3>
                <div class="text-content french">{pivot_intermediate.replace(chr(10), '<br>')}</div>
            </div>
            
            <div class="translation-block">
                <h3>Pivot Translation ({source_lang} → {pivot_lang} → {target_lang})</h3>
                <div class="text-content pivot">{pivot_final.replace(chr(10), '<br>')}</div>
            </div>
        </div>
        <hr>
        """
        return html
    
    def translate_epub_with_comparison(self, input_path: str, output_dir: str = "output", 
                                       mode: str = "both", source_lang: str = "English",
                                       pivot_lang: str = "French", target_lang: str = "Romanian"):
        """
        Translate EPUB using direct, pivot, or both methods
        
        Args:
            input_path: Path to input EPUB file
            output_dir: Directory for output files
            mode: "direct", "pivot", or "both" (default: "both")
            source_lang: Source language (default: "English")
            pivot_lang: Intermediate language for pivot translation (default: "French")
            target_lang: Target language (default: "Romanian")
        """
        if mode not in ["direct", "pivot", "both"]:
            raise ValueError("mode must be 'direct', 'pivot', or 'both'")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Reading EPUB from {input_path}...")
        book = epub.read_epub(input_path)
        chapters = self.extract_text_from_epub(input_path)
        
        print(f"Found {len(chapters)} chapters to translate")
        print(f"Translation mode: {mode.upper()}")
        print(f"Languages: {source_lang} → {target_lang} (direct), {source_lang} → {pivot_lang} → {target_lang} (pivot)\n")
        
        # Prepare output books based on mode
        direct_book = None
        pivot_book = None
        
        if mode in ["direct", "both"]:
            direct_book = self._create_book_template(book, f"Direct Translation ({source_lang} to {target_lang})")
        if mode in ["pivot", "both"]:
            pivot_book = self._create_book_template(book, f"Pivot Translation ({source_lang} to {target_lang} via {pivot_lang})")
        
        # HTML comparison document (only for "both" mode)
        comparison_html = None
        if mode == "both":
            comparison_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Translation Comparison</title>
                <style>
                    body {{ font-family: Georgia, serif; line-height: 1.6; max-width: 1400px; margin: 0 auto; padding: 20px; }}
                    h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                    h3 {{ color: #34495e; margin-top: 20px; }}
                    .chapter-comparison {{ margin-bottom: 50px; }}
                    .translation-block {{ margin: 20px 0; padding: 15px; border-radius: 5px; }}
                    .translation-block h3 {{ margin-top: 0; }}
                    .text-content {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #ccc; }}
                    .direct {{ border-left-color: #27ae60; }}
                    .french {{ border-left-color: #3498db; }}
                    .pivot {{ border-left-color: #e74c3c; }}
                    hr {{ margin: 40px 0; border: none; border-top: 2px dashed #ccc; }}
                </style>
            </head>
            <body>
                <h1>{source_lang} to {target_lang} Translation Comparison</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>Model:</strong> {self.model}</p>
                <p><strong>Method Comparison:</strong></p>
                <ul>
                    <li><span style="color: #27ae60;">●</span> <strong>Direct:</strong> {source_lang} → {target_lang} (single step)</li>
                    <li><span style="color: #e74c3c;">●</span> <strong>Pivot:</strong> {source_lang} → {pivot_lang} → {target_lang} (two steps)</li>
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
            original_paragraphs = chapter.get('paragraphs', [])
            
            direct_translation = None
            pivot_result = None
            
            # Direct translation
            if mode in ["direct", "both"]:
                print(f"Direct translation ({source_lang} → {target_lang})...")
                if original_paragraphs:
                    # Translate each paragraph separately for better quality
                    translated_paragraphs = []
                    for j, paragraph in enumerate(original_paragraphs):
                        if self.verbose:
                            print(f"  Translating paragraph {j+1}/{len(original_paragraphs)}")
                        if paragraph.strip():
                            translated_paragraphs.append(self._translate_chunk(paragraph, target_lang, source_lang))
                        else:
                            translated_paragraphs.append(paragraph)
                    direct_translation = '\n\n'.join(translated_paragraphs)
                else:
                    direct_translation = self.translate_direct(original_text, source_lang, target_lang)
            
            # Pivot translation
            if mode in ["pivot", "both"]:
                print(f"Pivot translation ({source_lang} → {pivot_lang} → {target_lang})...")
                if original_paragraphs:
                    # Translate each paragraph through pivot for better quality
                    intermediate_paragraphs = []
                    final_paragraphs = []
                    for j, paragraph in enumerate(original_paragraphs):
                        if self.verbose:
                            print(f"  Pivot translating paragraph {j+1}/{len(original_paragraphs)}")
                        if paragraph.strip():
                            # Source -> Pivot
                            intermediate = self._translate_chunk(paragraph, pivot_lang, source_lang)
                            intermediate_paragraphs.append(intermediate)
                            # Pivot -> Target
                            final = self._translate_chunk(intermediate, target_lang, pivot_lang)
                            final_paragraphs.append(final)
                        else:
                            intermediate_paragraphs.append(paragraph)
                            final_paragraphs.append(paragraph)
                    pivot_result = {
                        'intermediate': '\n\n'.join(intermediate_paragraphs),
                        'final': '\n\n'.join(final_paragraphs)
                    }
                else:
                    pivot_result = self.translate_pivot(original_text, source_lang, pivot_lang, target_lang)
            
            # Add to comparison HTML (only in "both" mode)
            if mode == "both":
                comparison_html += self.create_comparison_html(
                    i + 1, 
                    original_text, 
                    direct_translation,
                    pivot_result['intermediate'],
                    pivot_result['final'],
                    source_lang,
                    pivot_lang,
                    target_lang
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
                pivot_chapter.content = f'<html><body>{self._text_to_html(pivot_result["final"])}</body></html>'
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
        """Convert text to HTML paragraphs"""
        # First try to convert from Markdown, fallback to plain text
        if '#' in text or '- ' in text or '*' in text or '_' in text or '~' in text or '`' in text:
            return self._markdown_to_html(text)
        else:
            # Plain text conversion
            paragraphs = text.split('\n\n')
            html_paragraphs = [f'<p>{p.replace(chr(10), "<br/>")}</p>' for p in paragraphs if p.strip()]
            return '\n'.join(html_paragraphs)

def main():
    parser = argparse.ArgumentParser(description="BookLingua - Translate EPUB books using various AI models")
    parser.add_argument("input", help="Input EPUB file path")
    parser.add_argument("-o", "--output", default="output", help="Output directory (default: output)")
    parser.add_argument("-M", "--mode", choices=["direct", "pivot", "both"], default="direct",
                        help="Translation mode (default: direct)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-s", "--source-lang", default="English", help="Source language (default: English)")
    parser.add_argument("-p", "--pivot-lang", default="French", help="Pivot language (default: French)")
    parser.add_argument("-t", "--target-lang", default="Romanian", help="Target language (default: Romanian)")
    parser.add_argument("-u", "--base-url", help="Base URL for the API (e.g., https://api.openai.com/v1)")
    parser.add_argument("-m", "--model", default="gpt-4o", help="Model name to use (default: gpt-4o)")
    parser.add_argument("-k", "--api-key", help="API key for the translation service")
    
    # Preset configurations for common services
    parser.add_argument("--openai", action="store_true", help="Use OpenAI API")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama local server")
    parser.add_argument("--mistral", action="store_true", help="Use Mistral AI API")
    parser.add_argument("--deepseek", action="store_true", help="Use DeepSeek API")
    parser.add_argument("--lmstudio", action="store_true", help="Use LM Studio local server")
    parser.add_argument("--together", action="store_true", help="Use Together AI API")
    parser.add_argument("--openrouter", action="store_true", help="Use OpenRouter AI API")
    
    args = parser.parse_args()
    
    # Determine API configuration
    api_key = args.api_key
    base_url = args.base_url
    model = args.model
    
    # Handle preset configurations
    if args.openai:
        base_url = base_url or "https://api.openai.com/v1"
        model = model or "gpt-4o"
    elif args.ollama:
        base_url = base_url or "http://localhost:11434/v1"
        model = model or "qwen2.5:72b"
    elif args.mistral:
        base_url = base_url or "https://api.mistral.ai/v1"
        model = model or "mistral-large-latest"
        if not api_key:
            api_key = os.environ.get('MISTRAL_API_KEY')
    elif args.deepseek:
        base_url = base_url or "https://api.deepseek.com/v1"
        model = model or "deepseek-chat"
        if not api_key:
            api_key = os.environ.get('DEEPSEEK_API_KEY')
    elif args.lmstudio:
        base_url = base_url or "http://localhost:1234/v1"
        model = model or "qwen2.5-72b"
    elif args.together:
        base_url = base_url or "https://api.together.xyz/v1"
        model = model or "Qwen/Qwen2.5-72B-Instruct-Turbo"
        if not api_key:
            api_key = os.environ.get('TOGETHER_API_KEY')
    elif args.openrouter:
        base_url = base_url or "https://openrouter.ai/api/v1"
        model = model or "openai/gpt-4o"
        if not api_key:
            api_key = os.environ.get('OPENROUTER_API_KEY')
    
    # Use environment variable as fallback for API key
    if not api_key:
        api_key = os.environ.get('OPENAI_API_KEY')
    
    # Initialize translator
    translator = EPUBTranslator(
        api_key=api_key,
        base_url=base_url,
        model=model,
        verbose=args.verbose
    )
    
    # Run translation
    translator.translate_epub_with_comparison(
        input_path=args.input,
        output_dir=args.output,
        mode=args.mode,
        source_lang=args.source_lang,
        pivot_lang=args.pivot_lang,
        target_lang=args.target_lang
    )

# Usage examples
if __name__ == "__main__":
    main()
