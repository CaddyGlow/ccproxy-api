# Claude Code Proxy API Examples

This directory contains examples demonstrating how to use the Claude Code Proxy API with various features and formats.

## OpenAI Compatibility Examples

### 1. JSON Response Formats

#### Simple JSON Object
```bash
./openai_json_object_example.sh
```
Demonstrates using `response_format: {"type": "json_object"}` to get structured JSON responses.

#### JSON Schema
```bash
./openai_json_schema_example.sh
```
Shows how to use `response_format: {"type": "json_schema"}` with a defined schema to get precisely structured JSON responses.

#### Claude Code Options with OpenAI Format
```bash
./openai_claude_code_options_example.sh
```
Example of using the standard `/v1/chat/completions` endpoint with Claude Code specific options like `allowed_tools`, `permission_mode`, and `cwd`.

## Python Examples

### 2. Tool Usage Demonstrations

#### Anthropic Tools Demo
```bash
python anthropic_tools_demo.py
```
Python example showing how to use tools with the Anthropic API format.

#### OpenAI Tools Demo  
```bash
python openai_tools_demo.py
```
Python example demonstrating tool usage with OpenAI API format.

### 3. Interactive Applications

#### Textual Chat Agent
```bash
python textual_chat_agent.py
```
A terminal-based chat interface built with the Textual framework.

#### Simple Client
```bash
python client.py
```
Basic HTTP client example for making requests to the proxy API.

## Key Features Demonstrated

### OpenAI API Compatibility
- **Standard Endpoints**: `/v1/chat/completions` works as expected
- **Response Formats**: Support for `json_object` and `json_schema` response types
- **Model Mapping**: Automatic mapping from OpenAI model names to Claude models

### Claude Code Extensions
- **Tools Integration**: `allowed_tools` field for enabling specific tools
- **Permission Modes**: `permission_mode` for controlling edit permissions
- **Working Directory**: `cwd` field for setting execution context
- **Thinking Tokens**: `max_thinking_tokens` for reasoning capacity
- **System Prompt Extensions**: `append_system_prompt` for additional instructions

### Dual Path Architecture
- **SDK Path**: Used for requests with tools, uses OAuth authentication
- **Proxy Path**: Used for simple requests, requires API key authentication
- **Automatic Routing**: Hybrid service automatically chooses the appropriate path

## Authentication Notes

### For Tool-Enabled Requests (SDK Path)
- Uses OAuth authentication with Claude Code credentials
- Supports all Claude Code specific options
- Automatically detected when `allowed_tools` or similar fields are present

### For Simple Requests (Proxy Path)  
- Requires Anthropic API key authentication
- Standard OpenAI/Anthropic API compatibility
- Used when no Claude Code specific fields are detected

## Getting Started

1. **Start the server**:
   ```bash
   uv run python -m ccproxy.main
   ```

2. **Run any example**:
   ```bash
   cd examples
   ./openai_json_object_example.sh
   ```

3. **Modify for your needs**:
   - Change the `content` field for different prompts
   - Adjust `max_tokens` for longer/shorter responses
   - Add/remove tools in `allowed_tools` array
   - Modify JSON schemas for different structured outputs

## API Endpoints

- **OpenAI Format**: `/v1/chat/completions` (standard OpenAI compatibility)
- **OpenAI Format**: `/openai/v1/chat/completions` (explicit OpenAI namespace)
- **Anthropic Format**: `/v1/messages` (native Anthropic format)
- **Claude SDK**: `/claude-code/v1/*` (direct Claude SDK integration)

All examples work with the standard `/v1/chat/completions` endpoint for maximum compatibility.
