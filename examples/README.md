# OpenAI API Examples with Thinking Blocks

This directory contains examples demonstrating how to use the OpenAI-compatible API with Claude's thinking blocks feature through the proxy server.

## What are Thinking Blocks?

Thinking blocks are a special feature in Claude that captures the AI's reasoning process. They are formatted as:

```xml
<thinking signature="cryptographic_signature">
  The AI's internal reasoning process...
</thinking>
```

The proxy server preserves these blocks in multi-turn conversations, allowing you to:
- See how the AI reasons through problems
- Verify the authenticity of thinking blocks via signatures
- Maintain context across conversation turns

## Prerequisites

1. **Install dependencies:**
   ```bash
   pip install openai
   ```

2. **Set your API key:**
   ```bash
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   ```

3. **Start the proxy server:**
   ```bash
   uv run python main.py
   ```

## Examples

### 1. Simple Thinking Demo (`simple_thinking_demo.py`)

A minimal example showing basic thinking block functionality:

```bash
python examples/simple_thinking_demo.py
```

Features:
- Basic thinking block parsing
- Multi-turn conversation
- Token usage tracking

### 2. Advanced Demo (`openai_thinking_demo.py`)

Comprehensive examples with multiple scenarios:

```bash
python examples/openai_thinking_demo.py
```

Features:
- **Streaming mode**: See responses in real-time
- **Non-streaming mode**: Get complete responses
- **Tool use**: Thinking blocks with function calling
- **Multi-turn conversations**: Preserved thinking context

### 3. Structured Response Demo (`openai_structured_response_demo.py`)

Shows how to get structured JSON responses:

```bash
python examples/openai_structured_response_demo.py
```

## Key Concepts

### Model Mapping

The proxy automatically maps OpenAI models to Claude models with thinking enabled:

- `o1-preview` → Claude with high thinking budget
- `o1-mini` → Claude with medium thinking budget
- `gpt-4` → Standard Claude model

### Extracting Thinking Blocks

```python
import re

# Pattern to match thinking blocks
thinking_pattern = r'<thinking signature="([^"]*)">(.*?)</thinking>'
matches = re.findall(thinking_pattern, content, re.DOTALL)

# Remove thinking blocks to get visible content
visible_content = re.sub(thinking_pattern, '', content, flags=re.DOTALL)
```

### Multi-turn Conversations

Always include the full content (with thinking blocks) in the conversation history:

```python
messages = [
    {"role": "user", "content": "First question"},
    {"role": "assistant", "content": full_response_with_thinking},  # Important!
    {"role": "user", "content": "Follow-up question"}
]
```

## Tips

1. **Streaming**: When streaming, thinking blocks appear inline with the content
2. **Signatures**: Use signatures to verify thinking block authenticity
3. **Context**: Thinking blocks help maintain context across turns
4. **Models**: Use `o1-*` models to enable thinking features
5. **Temperature**: When using thinking mode, temperature must be set to 1.0

## Troubleshooting

If you get connection errors:
1. Ensure the proxy server is running
2. Check the base URL (default: `http://localhost:8000/openai/v1`)
3. Verify your API key is set correctly

If thinking blocks don't appear:
1. Use an appropriate model (`o1-preview`, `o1-mini`)
2. Check that your prompt encourages reasoning
3. Ensure you're parsing the content correctly
