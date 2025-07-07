# Textual Chat Agent

A simple chat agent using Anthropic SDK with streaming responses and a textual GUI with vim mode for input.

## Features

- **Streaming responses**: Real-time streaming of Claude's responses
- **Vim mode support**: ESC to enter vim mode, with basic vim commands
- **Rich text display**: Syntax highlighting and formatted output
- **Async handling**: Non-blocking UI during API calls

## Requirements

- Python 3.10+
- Anthropic API key
- Dependencies: `anthropic`, `textual`

## Setup

1. Set your Anthropic API key:
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```

2. Install dependencies (if not already installed):
   ```bash
   uv sync
   ```

3. Run the chat agent:
   ```bash
   python examples/textual_chat_agent.py
   ```

## Usage

### Input Modes

- **Insert Mode** (default): Type normally
- **Vim Mode**: Press `ESC` to enter vim mode

### Vim Commands

- `i` - Enter insert mode at cursor
- `a` - Enter insert mode after cursor
- `o` - Enter insert mode on new line
- `h` - Move cursor left
- `l` - Move cursor right
- `0` - Move to beginning of line
- `$` - Move to end of line
- `x` - Delete character at cursor

### Controls

- `Enter` - Send message
- `Ctrl+C` / `Ctrl+Q` - Quit application
- Click "Send" button - Alternative to Enter

## Example

```
Chat Agent - Ready to chat!
Use ESC to enter vim mode, then i/a/o to insert text.

You: Hello, how are you?

Claude: Hello! I'm doing well, thank you for asking. I'm here and ready to help with any questions or tasks you might have. How are you doing today?

You: What's the weather like?

Claude: I don't have access to real-time weather data, so I can't tell you the current weather conditions for your specific location. To get accurate weather information, I'd recommend:

1. Checking a weather website like Weather.com or AccuWeather
2. Using a weather app on your phone
3. Asking a voice assistant with internet access
4. Looking outside your window for current conditions

Is there anything else I can help you with?
```
