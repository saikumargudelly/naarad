# Naarad AI Assistant API Documentation

## Overview

The Naarad AI Assistant API provides a conversational interface powered by advanced language models. It supports text and image inputs, maintains conversation context, and offers various AI capabilities through specialized agents.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication for basic usage. However, rate limiting is applied per IP address.

## Rate Limiting

- **Limit**: 100 requests per minute per IP address
- **Headers**: Rate limit information is included in response headers
- **Exceeded**: Returns HTTP 429 (Too Many Requests) when limit is exceeded

## Endpoints

### 1. Health Check

**GET** `/api/health`

Check the health status of the service.

#### Response

```json
{
  "status": "healthy",
  "service": "naarad-chat",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### 2. Chat

**POST** `/api/chat`

Send a message to the AI assistant and receive a response.

#### Request Body

```json
{
  "message": "Hello, how are you?",
  "images": [],
  "conversation_id": "optional-conversation-id",
  "chat_history": []
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `message` | string | Yes | The user's message (1-5000 characters) |
| `images` | array | No | List of image URLs or base64 strings (max 5) |
| `conversation_id` | string | No | Unique identifier for conversation continuity |
| `chat_history` | array | No | Previous messages in the conversation (max 100) |

#### Response

```json
{
  "message": {
    "response": "Hello! I'm doing great, thank you for asking. How can I help you today?",
    "conversation_id": "conv_abc123",
    "metadata": {
      "agent_used": "responder",
      "model": "mixtral-8x7b-32768",
      "processing_time": "1.23s"
    }
  },
  "conversation_id": "conv_abc123",
  "sources": [],
  "processing_time": "1.23s"
}
```

#### Error Responses

**400 Bad Request**
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "message"],
      "msg": "Field required"
    }
  ]
}
```

**429 Too Many Requests**
```json
{
  "detail": "Rate limit exceeded"
}
```

**500 Internal Server Error**
```json
{
  "detail": "An error occurred while processing your request"
}
```

## Agent Types

The system uses different specialized agents based on the type of query:

### 1. Responder Agent
- **Purpose**: General conversation and responses
- **Use Cases**: Casual chat, greetings, general questions
- **Tools**: Vision analysis, basic responses

### 2. Researcher Agent
- **Purpose**: Information gathering and research
- **Use Cases**: Current events, factual questions, data retrieval
- **Tools**: Web search, information synthesis

### 3. Analyst Agent
- **Purpose**: Data analysis and insights
- **Use Cases**: Complex analysis, pattern recognition, detailed explanations
- **Tools**: Data processing, analytical reasoning

### 4. Quality Agent
- **Purpose**: Response quality assurance
- **Use Cases**: Fact-checking, accuracy verification, response improvement
- **Tools**: Quality assessment, fact verification

## Image Support

The API supports image analysis through the LLaVA vision model:

### Supported Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- PDF (.pdf)

### Image Input Methods
1. **Base64 Encoding**: Send images as base64 strings
2. **URLs**: Provide publicly accessible image URLs

### Example with Image

```json
{
  "message": "What's in this image?",
  "images": [
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
  ],
  "conversation_id": "conv_abc123"
}
```

## Conversation Management

### Conversation Continuity
- Use `conversation_id` to maintain context across multiple messages
- The system automatically generates conversation IDs if not provided
- Chat history is maintained for context-aware responses

### Chat History Format
```json
[
  {
    "role": "user",
    "content": "Hello, how are you?"
  },
  {
    "role": "assistant", 
    "content": "I'm doing great! How can I help you?"
  }
]
```

## Error Handling

### Common Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| 400 | Bad Request | Check request format and parameters |
| 422 | Validation Error | Verify input data meets requirements |
| 429 | Rate Limit Exceeded | Wait before making more requests |
| 500 | Internal Server Error | Contact support if persistent |

### Best Practices

1. **Rate Limiting**: Implement exponential backoff for retries
2. **Error Handling**: Always check response status codes
3. **Conversation Management**: Use conversation IDs for multi-turn chats
4. **Image Processing**: Compress images before sending to reduce payload size

## SDK Examples

### Python

```python
import requests

def chat_with_naarad(message, conversation_id=None):
    url = "http://localhost:8000/api/chat"
    payload = {
        "message": message,
        "conversation_id": conversation_id,
        "images": [],
        "chat_history": []
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    return response.json()

# Usage
result = chat_with_naarad("Hello, how are you?")
print(result["message"]["response"])
```

### JavaScript

```javascript
async function chatWithNaarad(message, conversationId = null) {
    const url = "http://localhost:8000/api/chat";
    const payload = {
        message: message,
        conversation_id: conversationId,
        images: [],
        chat_history: []
    };
    
    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
}

// Usage
chatWithNaarad("Hello, how are you?")
    .then(result => console.log(result.message.response))
    .catch(error => console.error('Error:', error));
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Application environment | `development` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `OPENROUTER_API_KEY` | OpenRouter API key | None |
| `TOGETHER_API_KEY` | Together AI API key | None |
| `BRAVE_API_KEY` | Brave Search API key | None |

### Model Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `DEFAULT_MODEL` | Default language model | `mistralai/Mixtral-8x7B-Instruct-v0.1` |
| `CHAT_MODEL` | Chat-specific model | `nousresearch/nous-hermes-2-mixtral-8x7b-dpo` |
| `EMBEDDING_MODEL` | Embedding model | `sentence-transformers/all-mpnet-base-v2` |

## Monitoring and Logging

### Log Levels
- **DEBUG**: Detailed debugging information
- **INFO**: General information about application flow
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages for failed operations
- **CRITICAL**: Critical errors that may cause system failure

### Performance Metrics
- Request processing time
- Agent selection statistics
- Error rates and types
- Rate limiting information

## Support

For technical support or questions about the API:

1. Check the [GitHub repository](https://github.com/your-repo/naarad)
2. Review the [FastAPI documentation](http://localhost:8000/docs)
3. Open an issue for bugs or feature requests

## Changelog

### Version 1.0.0
- Initial release
- Basic chat functionality
- Image analysis support
- Multi-agent system
- Rate limiting
- Conversation management 