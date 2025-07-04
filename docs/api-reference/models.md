# Models

Available Claude models for your local proxy.

## Supported Models

| Model ID | Description | Context Length | Use Case |
|----------|-------------|----------------|----------|
| `claude-sonnet-4-20250514` | Claude 4 Sonnet | 200K tokens | General purpose, latest model |
| `claude-3-5-sonnet-20241022` | Claude 3.5 Sonnet | 200K tokens | High-quality reasoning |
| `claude-3-7-sonnet-20250219` | Claude 3.7 Sonnet | 200K tokens | Enhanced capabilities |
| `claude-opus-4-20250514` | Claude 4 Opus | 200K tokens | Maximum capability |
| `claude-3-5-haiku-20241022` | Claude 3.5 Haiku | 200K tokens | Fast responses |

## Model Selection Guide

### For General Use
- **claude-sonnet-4-20250514**: Latest and most capable model
- **claude-3-5-sonnet-20241022**: Reliable choice for most tasks

### For Speed
- **claude-3-5-haiku-20241022**: Fastest response times
- **claude-3-7-sonnet-20250219**: Good balance of speed and quality

### For Complex Tasks
- **claude-opus-4-20250514**: Maximum reasoning capability
- **claude-sonnet-4-20250514**: Latest advanced features

## Model Capabilities

### All Models Support
- Text generation and analysis
- Code writing and debugging
- Mathematical reasoning
- Creative writing
- Language translation
- Summarization
- Question answering

### Advanced Features
- **Image Analysis**: Support for JPEG, PNG, GIF, WebP
- **Document Processing**: Text extraction and analysis
- **Code Understanding**: Multi-language code analysis
- **Streaming**: Real-time response streaming

## Usage Examples

### Model Selection in Requests

```json
{
  "model": "claude-sonnet-4-20250514",
  "messages": [
    {
      "role": "user",
      "content": "Write a Python function to calculate fibonacci numbers"
    }
  ],
  "max_tokens": 1000
}
```

### Dynamic Model Selection

```python
# Choose model based on task complexity
def get_model_for_task(task_type):
    if task_type == "simple_qa":
        return "claude-3-5-haiku-20241022"
    elif task_type == "code_generation":
        return "claude-3-5-sonnet-20241022"
    elif task_type == "complex_reasoning":
        return "claude-opus-4-20250514"
    else:
        return "claude-sonnet-4-20250514"  # Default to latest
```

## Model Availability

Models are available through your Claude subscription. The proxy automatically handles:
- Model validation
- Subscription checking
- Capability routing

## Performance Characteristics

| Model | Speed | Quality | Cost Efficiency |
|-------|-------|---------|-----------------|
| claude-3-5-haiku-20241022 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| claude-3-7-sonnet-20250219 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| claude-3-5-sonnet-20241022 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| claude-sonnet-4-20250514 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| claude-opus-4-20250514 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

## Error Handling

### Model Not Found

```json
{
  "error": {
    "type": "not_found_error",
    "message": "Model 'invalid-model' not found"
  }
}
```

### Model Unavailable

```json
{
  "error": {
    "type": "service_unavailable_error",
    "message": "Model 'claude-opus-4-20250514' temporarily unavailable"
  }
}
```

## Best Practices

1. **Start with Haiku** for testing and development
2. **Use Sonnet** for production applications
3. **Reserve Opus** for complex reasoning tasks
4. **Monitor usage** to optimize costs
5. **Cache responses** when appropriate

## Model Updates

The proxy automatically supports new Claude models as they become available. Check the `/v1/models` endpoint for the latest list of supported models.