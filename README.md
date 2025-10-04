# BookLingua

**Translate EPUB books using AI models. Supports direct, pivot, and comparison translations for high-quality multilingual book conversion.**

A Python tool for translating EPUB books using various AI models through their API endpoints. The tool supports multiple translation methods including direct translation and pivot translation (via an intermediate language) with side-by-side comparisons.

## Features

- Translate EPUB books between any languages using various AI models
- Support for multiple translation services (OpenAI, Ollama, Mistral, DeepSeek, Together AI, LM Studio, OpenRouter)
- Two translation methods:
  - **Direct**: Source → Target (single step)
  - **Pivot**: Source → Intermediate → Target (two steps)
- Side-by-side comparison of translation methods
- Preserves original formatting and structure
- Chunked translation for handling large texts
- HTML comparison output for evaluating translation quality

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

# Both translation methods with comparison
python booklingua.py input.epub --mode both
```

## Usage

### Translation Modes

```bash
# Direct translation only (default)
python booklingua.py input.epub --mode direct

# Pivot translation only  
python booklingua.py input.epub --mode pivot

# Both methods with comparison
python booklingua.py input.epub --mode both
```

### Language Configuration

```bash
# Set source, pivot, and target languages
python booklingua.py input.epub \
  --source-lang English \
  --pivot-lang French \
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

- `direct_translation.epub` - Direct translation (direct mode)
- `pivot_translation.epub` - Pivot translation (pivot mode)  
- `comparison.html` - Side-by-side comparison (both mode)

## How It Works

1. **Extract** text content from EPUB chapters
2. **Chunk** large content into manageable pieces
3. **Translate** using AI models with formatting preservation
4. **Reconstruct** translated content into complete chapters
5. **Generate** new EPUB files and comparison HTML

## Customization

Adjust parameters in the code:
- `chunk_size`: Max characters per request (default: 3000)
- `temperature`: Translation randomness (default: 0.3)
- `max_tokens`: Max tokens per response (default: 8000)

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
- `output/direct_translation.epub` - The translated book
- `output/` directory (if it doesn't exist)

#### Pivot Translation

Translate using French as an intermediate language:

```bash
python booklingua.py book.epub -M pivot -p French
```

This creates:
- `output/pivot_translation.epub` - The pivot-translated book

#### Both Methods with Comparison

Generate both direct and pivot translations with a comparison HTML:

```bash
python booklingua.py book.epub -M both -o translations
```

This creates:
- `translations/direct_translation.epub`
- `translations/pivot_translation.epub`
- `translations/comparison.html` - Side-by-side comparison

### Advanced Usage

#### Custom Languages

Translate from English to Spanish:

```bash
python booklingua.py book.epub -t Spanish -s English
```

Translate from French to German:

```bash
python booklingua.py book.epub -s French -t German -p English
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
python booklingua.py book.epub --openai -m gpt-4-turbo
```

#### Ollama (Local)

Use Ollama with local models:

```bash
python booklingua.py book.epub --ollama -m qwen2.5:72b
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
python booklingua.py book.epub -m gpt-3.5-turbo
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
    verbose=True
)

# Translate EPUB
translator.translate_epub_with_comparison(
    input_path="book.epub",
    output_dir="output",
    mode="both",
    source_lang="English",
    target_lang="Romanian"
)
```

#### Custom Languages

```python
# Translate from English to Spanish
translator.translate_epub_with_comparison(
    input_path="book.epub",
    mode="direct",
    source_lang="English",
    target_lang="Spanish"
)

# Translate using pivot method
translator.translate_epub_with_comparison(
    input_path="book.epub",
    mode="pivot",
    source_lang="English",
    pivot_lang="French",
    target_lang="German"
)
```

#### Individual Text Translation

```python
# Translate individual text
result = translator.translate_direct(
    "Hello, how are you?",
    source_lang="English",
    target_lang="Romanian"
)
print(result)  # "Salut, cum ești?"

# Use pivot translation for individual text
pivot_result = translator.translate_pivot(
    "Hello, how are you?",
    source_lang="English",
    pivot_lang="French",
    target_lang="Romanian"
)
print(pivot_result['intermediate'])  # "Bonjour, comment allez-vous?"
print(pivot_result['final'])  # "Salut, cum ești?"
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
- Use pivot translation with `-M pivot`
- Enable verbose mode with `-v` to see intermediate results

**Slow Translation**
- Reduce verbose output
- Use smaller chunk sizes (modify source code)
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
4. **Compare different methods** using `-M both` for quality assessment
5. **Use appropriate models** for your content type (technical vs. literary)
6. **Set environment variables** for API keys to avoid exposing them in command history
7. **Customize pivot language** based on source-target language pair for best results

### Integration Examples

#### With Shell Scripts

```bash
#!/bin/bash
# batch_translate.sh

for book in *.epub; do
    echo "Translating $book..."
    python booklingua.py "$book" -M both -o "translations_${book%.epub}"
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
        translator.translate_epub_with_comparison(
            input_path=filename,
            output_dir="output",
            mode="both"
        )
```

## Contributing

Contributions welcome! Please submit Pull Requests.
