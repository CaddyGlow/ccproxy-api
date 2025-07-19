#!/usr/bin/env python3
"""Simple demo of thinking blocks with OpenAI client.

This is a minimal example showing how thinking blocks work.
"""

import os
import re

from openai import OpenAI


def main():
    # Initialize the client
    client = OpenAI(
        base_url="http://localhost:8000/openai/v1",
        api_key=os.getenv("ANTHROPIC_API_KEY", "your-api-key-here"),
    )

    print("Simple Thinking Blocks Demo")
    print("==========================\n")

    # Create a conversation
    messages = [
        {
            "role": "user",
            "content": "What's the most efficient way to sort a list of 10 numbers? Think through the options.",
        }
    ]

    print(f"User: {messages[0]['content']}\n")

    # Get response with thinking
    response = client.chat.completions.create(
        model="o1-mini",  # This enables thinking
        messages=messages,
        temperature=1.0,  # Must be 1.0 for thinking mode
    )

    # Extract the full content
    content = response.choices[0].message.content

    # Parse thinking blocks
    thinking_pattern = r'<thinking signature="([^"]*)">(.*?)</thinking>'
    thinking_matches = re.findall(thinking_pattern, content, re.DOTALL)

    # Extract visible content (without thinking blocks)
    visible_content = re.sub(thinking_pattern, "", content, flags=re.DOTALL).strip()

    # Display results
    print("Assistant's Response:")
    print("-" * 50)
    print(visible_content)
    print("-" * 50)

    if thinking_matches:
        print("\n[THINKING PROCESS]")
        for i, (signature, thinking) in enumerate(thinking_matches, 1):
            print(f"\nThinking Block {i}:")
            print(f"Signature: {signature}")
            print(f"Content: {thinking.strip()}")

    # Continue the conversation
    messages.append(
        {
            "role": "assistant",
            "content": content,  # Include full content with thinking blocks
        }
    )
    messages.append(
        {
            "role": "user",
            "content": "What about if the list had 1 million numbers instead?",
        }
    )

    print(f"\n\nUser: {messages[-1]['content']}\n")

    # Get second response
    response2 = client.chat.completions.create(
        model="o1-mini",
        messages=messages,
        temperature=1.0,  # Must be 1.0 for thinking mode
    )

    content2 = response2.choices[0].message.content

    # Parse second response
    thinking_matches2 = re.findall(thinking_pattern, content2, re.DOTALL)
    visible_content2 = re.sub(thinking_pattern, "", content2, flags=re.DOTALL).strip()

    print("Assistant's Response:")
    print("-" * 50)
    print(visible_content2)
    print("-" * 50)

    if thinking_matches2:
        print("\n[THINKING PROCESS]")
        for i, (signature, thinking) in enumerate(thinking_matches2, 1):
            print(f"\nThinking Block {i}:")
            print(f"Signature: {signature}")
            print(f"Content: {thinking.strip()}")

    # Show usage stats
    print("\n\nToken Usage:")
    print(f"First response: {response.usage.total_tokens} tokens")
    print(f"Second response: {response2.usage.total_tokens} tokens")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("1. The proxy server is running: uv run python main.py")
        print("2. Set ANTHROPIC_API_KEY environment variable")
        print("3. Install openai: pip install openai")
