from openai import OpenAI

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://172.16.78.10:33561/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen2-VL-72B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://www.hubspot.com/hubfs/assets/hubspot.com/web-team/WBZ/Feature%20Pages/website-drag-and-drop/custom-website-en.webp"
                    },
                },
                {"type": "text", "text": "Where is the button with a `Image` logo on it?"},
            ],
        },
    ],
)
print("Chat response:", chat_response)