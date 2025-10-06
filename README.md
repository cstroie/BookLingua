# BookLingua

**Translate EPUB books using AI models with database caching and context preservation.**

A Python tool for translating EPUB books using various AI models through their API endpoints. The tool focuses on direct translation with advanced features like database caching, context management, and quality assessment.

## Features

- Translate EPUB books between any languages using various AI models
- Support for multiple translation services (OpenAI, Ollama, Mistral, DeepSeek, Together AI, LM Studio, OpenRouter)
- Direct translation method (Source → Target)
- Preserves original formatting and structure
- Chunked translation for handling large texts
- Database caching for reliability and resume capability
- Context preservation for consistency across chapters
- Quality assessment with fluency scoring
- Progress tracking with timing statistics

## Installation

1. Install dependencies:
```bash
pip install ebooklib beautifulsoup4 requests
```

2. Set your API key (optional):
```bash
export OPENAI_API_KEY=your_api_key_here
```

## Quick Start

```bash
# Basic translation
python booklingua.py input.epub

# Custom languages
python booklingua.py input.epub -s German -t Spanish

# Verbose mode
python booklingua.py input.epub --verbose
```

## Usage

### Basic Translation

```bash
# Direct translation with default settings
python booklingua.py input.epub

# Custom languages
python booklingua.py input.epub -s German -t Spanish

# Verbose mode for detailed progress
python booklingua.py input.epub --verbose
```

### Language Configuration

```bash
# Set source and target languages
python booklingua.py input.epub \
  --source-lang English \
  --target-lang Romanian
```

### API Services

```bash
# OpenAI
python booklingua.py input.epub --openai -k YOUR_KEY

# Ollama (local)
python booklingua.py input.epub --ollama

# Mistral AI
python booklingua.py input.epub --mistral -k YOUR_KEY

# DeepSeek
python booklingua.py input.epub --deepseek -k YOUR_KEY

# Together AI
python booklingua.py input.epub --together -k YOUR_KEY

# LM Studio (local)
python booklingua.py input.epub --lmstudio

# OpenRouter
python booklingua.py input.epub --openrouter -k YOUR_KEY
```

### Manual Configuration

```bash
python booklingua.py input.epub \
  --base-url https://api.openai.com/v1 \
  --api-key YOUR_KEY \
  --model gpt-4o
```

## Supported Services

| Service      | Flag        | Default Model                     | Default URL                     |
|--------------|-------------|-----------------------------------|---------------------------------|
| OpenAI       | `--openai`  | `gpt-4o`                          | https://api.openai.com/v1       |
| Ollama       | `--ollama`  | `qwen2.5:72b`                     | http://localhost:11434/v1       |
| Mistral AI   | `--mistral` | `mistral-large-latest`           | https://api.mistral.ai/v1       |
| DeepSeek     | `--deepseek`| `deepseek-chat`                  | https://api.deepseek.com/v1     |
| Together AI  | `--together`| `Qwen/Qwen2.5-72B-Instruct-Turbo`| https://api.together.xyz/v1     |
| LM Studio    | `--lmstudio`| `qwen2.5:72b`                     | http://localhost:1234/v1        |
| OpenRouter   | `--openrouter`| `openai/gpt-4o`                 | https://openrouter.ai/api/v1    |

## Output Files

- `translated.epub` - The translated book
- `book.db` - SQLite database with cached translations (same name as input EPUB)

## How It Works

1. **Extract** text content from EPUB chapters
2. **Initialize** SQLite database for caching translations
3. **Prefill** context with existing translations
4. **Translate** each chapter with progress tracking
5. **Reconstruct** translated content into complete chapters
6. **Generate** new EPUB file with preserved structure

## Customization

Adjust parameters in the code:
- `chunk_size`: Max characters per request (default: 3000)
- `temperature`: Translation randomness (default: 0.5)
- `max_tokens`: Max tokens per response (default: 4096)

### Database Caching

BookLingua uses SQLite database caching to improve reliability and performance:

- **Database Location**: Creates a `.db` file with the same name as your input EPUB (e.g., `book.epub` → `book.db`)
- **Caching Strategy**: Stores all translations with source language, target language, and source text as unique keys
- **Benefits**: 
  - Resume interrupted translations
  - Avoid re-translating identical content
  - Faster subsequent translations of the same content
  - Track translation progress and statistics
- **Automatic**: Enabled by default when an EPUB path is provided
- **Persistence**: Database remains after translation for future use

### Context Management

The tool manages translation context to maintain consistency:

- **Context Window**: Maintains the last 5 translation exchanges for continuity
- **Benefits**: 
  - Consistent terminology across chapters
  - Better handling of repeated phrases
  - Improved coherence in long documents
- **Chapter-based Reset**: Context is reset between chapters to prevent drift
- **Prefill Strategy**: Uses existing translations and random paragraphs to initialize context

### Quality Assessment

The tool includes built-in quality assessment features:

- **Fluency Scoring**: Evaluates translation quality based on linguistic patterns
- **Progress Tracking**: Shows real-time statistics and estimated completion times
- **Error Detection**: Identifies common translation issues

## License

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

## Usage Examples

### Basic Usage

#### Direct Translation

Translate an EPUB book directly from English to Romanian:

```bash
python booklingua.py book.epub
```

This will create:
- `output/translated.epub` - The translated book
- `output/book.db` - SQLite database with cached translations
- `output/` directory (if it doesn't exist)

### Advanced Usage

#### Custom Languages

Translate from English to Spanish:

```bash
python booklingua.py book.epub -t Spanish -s English
```

#### Verbose Output

Enable detailed progress reporting:

```bash
python booklingua.py book.epub -v
```

#### Custom Output Directory

Save translations to a specific directory:

```bash
python booklingua.py book.epub -o my_translations
```

### AI Provider Presets

#### OpenAI

Use OpenAI's GPT models:

```bash
python booklingua.py book.epub --openai -m gpt-4o
```

#### Ollama (Local)

Use Ollama with local models:

```bash
python booklingua.py book.epub --ollama -m gemma3n:e4b
```

#### Mistral AI

Use Mistral AI's models:

```bash
python booklingua.py book.epub --mistral -m mistral-large-latest
```

#### DeepSeek

Use DeepSeek's models:

```bash
python booklingua.py book.epub --deepseek -m deepseek-chat
```

#### LM Studio (Local)

Use LM Studio with local models:

```bash
python booklingua.py book.epub --lmstudio -m qwen2.5-72b
```

#### Together AI

Use Together AI's models:

```bash
python booklingua.py book.epub --together -m Qwen/Qwen2.5-72B-Instruct-Turbo
```

#### OpenRouter

Use OpenRouter with various models:

```bash
python booklingua.py book.epub --openrouter -m openai/gpt-4o
```

### Custom API Configuration

#### Custom Endpoint

Use a custom API endpoint:

```bash
python booklingua.py book.epub -u https://api.example.com/v1 -k your-api-key
```

#### Custom Model

Use a specific model with default OpenAI endpoint:

```bash
python booklingua.py book.epub -m gpt-4o
```

### Environment Variables

Set API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export MISTRAL_API_KEY="your-mistral-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export TOGETHER_API_KEY="your-together-api-key"
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

Then run without `-k` option:

```bash
python booklingua.py book.epub --mistral
```

### Python API Usage

#### Basic Translation

```python
from booklingua import EPUBTranslator

# Initialize translator
translator = EPUBTranslator(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    model="gpt-4o",
    verbose=True,
    epub_path="book.epub"
)

# Translate EPUB
translator.translate_epub(
    input_path="book.epub",
    output_dir="output",
    source_lang="English",
    target_lang="Romanian"
)
```

#### Custom Languages

```python
# Translate from English to Spanish
translator.translate_epub(
    input_path="book.epub",
    source_lang="English",
    target_lang="Spanish"
)
```

### Troubleshooting

#### Common Issues

**API Key Not Found**
Set the API key using `-k` option or environment variable.

**Model Not Available**
Check if the model name is correct for your provider.

**Connection Errors**
Verify your internet connection and API endpoint URL.

**Permission Errors**
Ensure you have write permissions for the output directory.

#### Quality Issues

**Poor Translation Quality**
- Try different models using `-m` option
- Enable verbose mode with `-v` to see intermediate results

**Slow Translation**
- Reduce verbose output
- Use a faster model or local server

#### Error Messages

**"API request failed"**
- Check API key and endpoint
- Verify model availability
- Check network connectivity

**"File not found"**
- Verify input file path
- Check file permissions

### Best Practices

1. **Always backup your original EPUB files** before translation
2. **Use verbose mode** (`-v`) for the first translation to monitor progress
3. **Test with small sections** first before translating entire books
4. **Use appropriate models** for your content type (technical vs. literary)
5. **Set environment variables** for API keys to avoid exposing them in command history

### Integration Examples

#### With Shell Scripts

```bash
#!/bin/bash
# batch_translate.sh

for book in *.epub; do
    echo "Translating $book..."
    python booklingua.py "$book" -o "translations_${book%.epub}"
done
```

#### With Python Scripts

```python
#!/usr/bin/env python3
# batch_translate.py

import os
from booklingua import EPUBTranslator

translator = EPUBTranslator(api_key="your-api-key")

for filename in os.listdir("."):
    if filename.endswith(".epub"):
        print(f"Processing {filename}...")
        translator.translate_epub(
            input_path=filename,
            output_dir="output"
        )
```

## Contributing

Contributions welcome! Please submit Pull Requests.
