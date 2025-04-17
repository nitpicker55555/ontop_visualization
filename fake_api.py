import json
import re
import os
from typing import Dict, List, Union, Any

from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("hub_api_key")
os.environ['OPENAI_BASE_URL'] = "https://api.openai-hub.com/v1"
client = OpenAI()


def message_template(role: str, content: str) -> Dict[str, str]:
    """Create a message template dictionary.

    Args:
    role: message role ('system', 'user', or 'assistant')
    content: message content

    Returns:
    dictionary containing role and content
    """
    return {'role': role, 'content': content}


@retry(wait=wait_random_exponential(multiplier=1, max=40),
       stop=stop_after_attempt(3))
def chat_single(messages: List[Dict[str, str]],
                mode: str = "",
                model: str = 'gpt-4o',
                temperature: float = 0,
                verbose: bool = False):
    """Send a single chat request to the OpenAI API.

    Args:
    messages: list of messages
    mode: response mode ('stream', 'json', 'json_few_shot', or empty string for normal mode)
    model: model to use
    temperature: temperature parameter, controls response randomness
    verbose: whether to print verbose information

    Returns:
    Returns different types of responses depending on the mode
    """
    if mode == "stream":
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
            max_tokens=2560
        )
        return response
    elif mode == "json":
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            temperature=temperature,
            messages=messages
        )
        return response.choices[0].message.content
    elif mode == 'json_few_shot':
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=2560
        )
        result = response.choices[0].message.content
        if verbose:
            print(result)
        return extract_json_and_similar_words(result)
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content


def format_list_string(input_str: str) -> str:
    """Format a string containing a list into valid JSON.

    Args:
    input_str: string containing a list

    Returns:
    Formatted JSON string
    """
    match = re.search(r'\{\s*"[^"]+"\s*:\s*\[(.*?)\]\s*\}', input_str)
    if not match:
        return "Invalid input format"

    list_content = match.group(1)
    elements = [e.strip() for e in list_content.split(',')]

    formatted_elements = []
    for elem in elements:
        if not re.match(r'^([\'"])(.*)\1$', elem):
            elem = f'"{elem}"'
        formatted_elements.append(elem)

    return f'{{ "similar_words":[{", ".join(formatted_elements)}]}}'


def extract_json_and_similar_words(text: str) -> Dict[str, Any]:
    """Extract JSON data from text.

    Args:
    text: Text containing JSON data

    Returns:
    Dictionary of extracted JSON data
    """
    try:
        json_match = re.search(r'```json\s*({.*?})\s*```', text, re.DOTALL)

        if not json_match:
            raise ValueError("No JSON data found in the text.")

        json_str = json_match.group(1)
        if 'similar_words' in text:
            data = json.loads(format_list_string(json_str))
        else:
            data = json.loads(json_str)

        return data
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        return {"error": str(e)}


def run_examples():
    """Run all mode examples to show different API call methods."""

    # Basic message template, used for all examples
    base_messages = [
        message_template('system',
                         'hi'),
    ]

    print("\n===== 1. Standard mode example =====")
    standard_messages = base_messages.copy()
    standard_messages.append(
        message_template('user', 'Who are you'))

    standard_response = chat_single(standard_messages)
    print(f"Response:\n{standard_response}\n")

    print("\n===== 2. Streaming response mode example =====")
    stream_messages = base_messages.copy()
    stream_messages.append(
        message_template('user',
                         'Explain the concept of asynchronous programming in Python.'))
    stream_response = chat_single(stream_messages, mode="stream")

    collected_response = ""
    print("Streaming response:")
    for chunk in stream_response:
        if chunk.choices[0].delta.content is not None:
            content_chunk = chunk.choices[0].delta.content
            collected_response += content_chunk
            print(content_chunk, end="", flush=True)

    print("\n\nFull collected response:")
    print(collected_response)

    print("\n===== 3. JSON response mode example =====")
    json_messages = base_messages.copy()
    json_messages.append(message_template('user',
                                          'Provide the names and brief descriptions of the three main Python data structures in JSON format.'))
    json_response = chat_single(json_messages, mode="json")
    print(f"JSON response:\n{json_response}\n")
    print(f"parsed JSON:\n{json.loads(json_response)}\n")
    print(
        "\n===== 4. JSON Few-Shot Examples =====")  # The reasoning part can be retained to reduce the performance degradation caused by the structure output text
    few_shot_messages = base_messages.copy()
    few_shot_messages.append(message_template('user',
                                              '''Give words similar to"programming"。

                                              response in json:
                                              ```json
                                              {
                                                "similar_words": ["coding", "development", ...]
                                              }
                                              ```
                                              '''))

    few_shot_response = chat_single(few_shot_messages, mode="json_few_shot",
                                    verbose=True)
    print(f"Final response:\n{few_shot_response}\n")


if __name__ == "__main__":
    run_examples()