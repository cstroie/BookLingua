# EPUB Translator

A Python tool for translating EPUB books using various AI models through their API endpoints. The tool supports multiple translation methods including direct translation and pivot translation (via an intermediate language) with side-by-side comparisons. Languages are configurable via command line options.

## Features

- Translate EPUB books between any languages using various AI models
- Support for multiple translation services (OpenAI, Ollama, Mistral, DeepSeek, Together AI, LM Studio)
- Two translation methods:
  - **Direct**: Source → Target (single step)
  - **Pivot**: Source → Intermediate → Target (two steps)
- Side-by-side comparison of translation methods
- Preserves original formatting and structure
- Chunked translation for handling large texts
- HTML comparison output for evaluating translation quality
- Configurable source, pivot, and target languages

## Requirements

- Python 3.7+
- ebooklib
- beautifulsoup4
- requests

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd epub-translator
```

2. Install dependencies:
```bash
pip install ebooklib beautifulsoup4 requests
```

## Usage

### Basic Usage

```bash
python epub_translator.py input.epub
```

This will translate the EPUB file using both direct and pivot methods with default languages (English → Romanian) and generate:
- `direct_translation.epub` - Direct translation
- `pivot_translation.epub` - Pivot translation
- `comparison.html` - Side-by-side comparison

### Verbose Mode

To see each chunk being translated along with its translation:

```bash
python epub_translator.py input.epub --verbose
```

### Custom Languages

```bash
python epub_translator.py input.epub \
  --source-lang German \
  --pivot-lang English \
  --target-lang Spanish
```

Or using short options:
```bash
python epub_translator.py input.epub -s German -p English -t Spanish
```

### Specify Output Directory

```bash
python epub_translator.py input.epub -o /path/to/output
```

### Translation Modes

Choose a specific translation mode:
```bash
# Direct translation only
python epub_translator.py input.epub --mode direct

# Pivot translation only
python epub_translator.py input.epub --mode pivot

# Both methods (default)
python epub_translator.py input.epub --mode both
```

### API Configuration

#### Using Preset Configurations

```bash
# OpenAI
python epub_translator.py input.epub --openai --api-key YOUR_API_KEY

# Ollama (local)
python epub_translator.py input.epub --ollama

# Mistral AI
python epub_translator.py input.epub --mistral --api-key YOUR_API_KEY

# DeepSeek
python epub_translator.py input.epub --deepseek --api-key YOUR_API_KEY

# Together AI
python epub_translator.py input.epub --together --api-key YOUR_API_KEY

# LM Studio (local)
python epub_translator.py input.epub --lmstudio
```

#### Manual Configuration

```bash
python epub_translator.py input.epub \
  --base-url https://api.openai.com/v1 \
  --api-key YOUR_API_KEY \
  --model gpt-4o
```

Or using short options:
```bash
python epub_translator.py input.epub -u https://api.openai.com/v1 --api-key YOUR_API_KEY -m gpt-4o
```

### Environment Variables

You can also set API keys as environment variables:
```bash
export OPENAI_API_KEY=your_openai_key
export MISTRAL_API_KEY=your_mistral_key
export DEEPSEEK_API_KEY=your_deepseek_key
export TOGETHER_API_KEY=your_together_key
```

## Supported Services

| Service      | Flag        | Default Model     | Default URL              |
|--------------|-------------|-------------------|--------------------------|
| OpenAI       | `--openai`  | `gpt-4o`          | https://api.openai.com/v1 |
| Ollama       | `--ollama`  | `qwen2.5:72b`     | http://localhost:11434/v1 |
| Mistral AI   | `--mistral` | `mistral-large-latest` | https://api.mistral.ai/v1 |
| DeepSeek     | `--deepseek`| `deepseek-chat`   | https://api.deepseek.com/v1 |
| Together AI  | `--together`| `Qwen/Qwen2.5-72B-Instruct-Turbo` | https://api.together.xyz/v1 |
| LM Studio    | `--lmstudio`| `qwen2.5-72b`     | http://localhost:1234/v1 |

## Output Files

When using `--mode both` (default), the tool generates:
- `direct_translation.epub`: Book translated directly from source to target language
- `pivot_translation.epub`: Book translated via intermediate language
- `comparison.html`: Side-by-side comparison of original text, direct translation, and pivot translation

## How It Works

1. **Text Extraction**: The tool extracts text content from each chapter of the EPUB file
2. **Chunking**: Large chapters are split into manageable chunks at paragraph boundaries
3. **Translation**: Each chunk is sent to the AI model with specific instructions to maintain formatting
4. **Reconstruction**: Translated chunks are reassembled into complete chapters
5. **EPUB Generation**: New EPUB files are created with the translated content

## Customization

You can adjust translation parameters in the code:
- `chunk_size`: Maximum characters per translation request (default: 3000)
- `temperature`: Controls randomness in translation (default: 0.3)
- `max_tokens`: Maximum tokens per response (default: 8000)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
