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

## Contributing

Contributions welcome! Please submit Pull Requests.
