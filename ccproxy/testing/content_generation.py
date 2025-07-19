"""Content generation utilities for testing requests and responses."""

import random
from typing import Any

from ccproxy.testing.config import RequestScenario


class MessageContentGenerator:
    """Generate realistic message content for testing."""

    def __init__(self) -> None:
        self.response_templates = self._load_response_templates()
        self.request_templates = self._load_request_templates()

    def _load_response_templates(self) -> dict[str, list[str]]:
        """Load variety of response templates."""
        return {
            "short": [
                "Hello! How can I help you today?",
                "I'm happy to assist you.",
                "What would you like to know?",
                "I'm here to help!",
                "How may I assist you?",
            ],
            "medium": [
                "I'd be happy to help you with that. Let me provide you with some information that should be useful for your question.",
                "That's an interesting question. Here's what I can tell you about this topic based on my knowledge.",
                "I understand what you're asking about. Let me break this down into a clear explanation for you.",
            ],
            "long": [
                "This is a comprehensive topic that requires a detailed explanation. Let me walk you through the key concepts step by step. First, it's important to understand the foundational principles. Then we can explore the more advanced aspects. Finally, I'll provide some practical examples to illustrate the concepts.",
                "That's an excellent question that touches on several important areas. To give you a complete answer, I need to cover multiple aspects. Let me start with the basic framework, then dive into the specifics, and conclude with some recommendations based on best practices in this field.",
            ],
            "tool_use": [
                "I'll help you with that calculation.",
                "Let me solve that mathematical problem for you.",
                "I can compute that result using the calculator tool.",
            ],
        }

    def _load_request_templates(self) -> dict[str, list[str]]:
        """Load variety of request message templates."""
        return {
            "short": [
                "Hello!",
                "How are you?",
                "What's the weather like?",
                "Tell me a joke.",
                "What time is it?",
            ],
            "long": [
                "I need help writing a detailed technical document about API design patterns. Can you provide a comprehensive guide covering REST principles, authentication methods, error handling, and best practices for scalable API development?",
                "Please explain the differences between various machine learning algorithms including supervised learning, unsupervised learning, and reinforcement learning. Include examples of when to use each approach and their respective advantages and disadvantages.",
                "I'm planning a complex software architecture for a distributed system. Can you help me understand microservices patterns, database sharding strategies, caching layers, and how to handle eventual consistency in distributed transactions?",
            ],
            "tool_use": [
                "Calculate 23 * 45 + 67 for me",
                "What's the result of (150 / 3) * 2.5?",
                "Help me calculate the compound interest on $1000 at 5% for 3 years",
            ],
        }

    def get_request_message_content(self, message_type: str) -> str:
        """Get request message content based on type."""
        if message_type in self.request_templates:
            return random.choice(self.request_templates[message_type])
        else:
            # Fallback to short message for unknown types
            return random.choice(self.request_templates["short"])

    def get_response_content(
        self, message_type: str, model: str
    ) -> tuple[str, int, int]:
        """Generate response content with realistic token counts."""
        # Select base template
        if message_type == "tool_use":
            base_content = random.choice(self.response_templates["tool_use"])
            # Add calculation result
            result = random.randint(1, 1000)
            content = f"{base_content} The result is {result}."
        elif message_type in self.response_templates:
            content = random.choice(self.response_templates[message_type])
        else:
            # Mix of different lengths for unknown types
            template_type = random.choice(["short", "medium", "long"])
            content = random.choice(self.response_templates[template_type])

        # Calculate realistic token counts based on content
        # Rough estimate: ~4 characters per token
        estimated_output_tokens = max(1, len(content) // 4)

        # Add some randomness but keep it realistic
        output_tokens = random.randint(
            max(1, estimated_output_tokens - 10), estimated_output_tokens + 20
        )

        # Input tokens based on typical request sizes (10-500 range)
        input_tokens = random.randint(10, 500)

        return content, input_tokens, output_tokens


class PayloadBuilder:
    """Build request payloads for different API formats."""

    def __init__(self) -> None:
        self.content_generator = MessageContentGenerator()

    def build_anthropic_payload(self, scenario: RequestScenario) -> dict[str, Any]:
        """Build Anthropic format payload."""
        payload = {
            "model": scenario.model,
            "messages": [
                {
                    "role": "user",
                    "content": self.content_generator.get_request_message_content(
                        scenario.message_type
                    ),
                }
            ],
            "stream": scenario.streaming,
            "max_tokens": random.randint(100, 4000),  # Realistic token limits
        }

        if scenario.message_type == "tool_use":
            payload["tools"] = [
                {
                    "name": "calculator",
                    "description": "Perform basic calculations",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Math expression to evaluate",
                            }
                        },
                        "required": ["expression"],
                    },
                }
            ]

        return payload

    def build_openai_payload(self, scenario: RequestScenario) -> dict[str, Any]:
        """Build OpenAI format payload."""
        messages = [
            {
                "role": "user",
                "content": self.content_generator.get_request_message_content(
                    scenario.message_type
                ),
            }
        ]

        payload = {
            "model": scenario.model,
            "messages": messages,
            "stream": scenario.streaming,
            "max_tokens": random.randint(100, 4000),  # Realistic token limits
        }

        if scenario.message_type == "tool_use":
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "description": "Perform basic calculations",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "Math expression to evaluate",
                                }
                            },
                            "required": ["expression"],
                        },
                    },
                }
            ]

        return payload

    def build_payload(self, scenario: RequestScenario) -> dict[str, Any]:
        """Build request payload based on scenario format."""
        # Use custom payload if provided
        if scenario.custom_payload:
            return scenario.custom_payload

        # Build format-specific payload
        if scenario.api_format == "openai":
            return self.build_openai_payload(scenario)
        else:
            return self.build_anthropic_payload(scenario)
