"""Tools to generate from OpenAI prompts.
Adopted from https://github.com/zeno-ml/zeno-build/"""

import asyncio
import logging
import os
import random
import time
import pdb
from typing import Any, Union
import requests

import aiolimiter
import openai
from openai import AsyncOpenAI, OpenAI
from tqdm.asyncio import tqdm_asyncio
from openai import AzureOpenAI

if "OPENAI_API_BASE" not in os.environ:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    aclient = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
else:
    # Used for running vllm models.
    print("WARNING: Using OPENAI_API_KEY=EMPTY")
    client = OpenAI(
        api_key="EMPTY", base_url=os.environ["OPENAI_API_BASE"]
    )
    aclient = AsyncOpenAI(
        api_key="EMPTY", base_url=os.environ["OPENAI_API_BASE"]
    )


def retry_with_exponential_backoff(  # type: ignore
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 5,
        errors: tuple[Any] = (
                openai.RateLimitError,
                openai.BadRequestError,
                openai.InternalServerError,
        ),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:

                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


async def _throttled_openai_completion_acreate(
        engine: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(3):
            try:
                return await aclient.completions.create(
                    engine=engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(10)
            except openai.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return {"choices": [{"message": {"content": ""}}]}


async def agenerate_from_openai_completion(
        prompts: list[str],
        engine: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        context_length: int,
        requests_per_minute: int = 300,
) -> list[str]:
    """Generate from OpenAI Completion API.

    Args:
        prompts: list of prompts
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_completion_acreate(
            engine=engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for prompt in prompts
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    return [x["choices"][0]["text"] for x in responses]


@retry_with_exponential_backoff
def generate_from_openai_completion(
        prompt: str,
        engine: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        context_length: int,
        stop_token: str | None = None,
        num_outputs: int = 1,
) -> str:
    openai_api_key = "EMPTY"
    openai_api_base = os.environ["OPENAI_API_BASE"]
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    models = client.models.list()
    model = models.data[0].id
    assert isinstance(prompt, str)
    completion = client.completions.create(model=model,
                                           prompt=prompt,
                                           stop=[stop_token],
                                           n=num_outputs,
                                           temperature=temperature,
                                           max_tokens=max_tokens,
                                           top_p=top_p)
    if num_outputs > 1:
        answer: list[str] = [completion.choices[i].text.strip() for i in range(len(completion.choices))]
    else:
        answer: str = completion.choices[0].text.strip()
    print("Answer:", answer)
    return answer


async def _throttled_openai_chat_completion_acreate(
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(3):
            try:
                return await aclient.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(10)
            except asyncio.exceptions.TimeoutError:
                logging.warning("OpenAI API timeout. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
            except openai.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return {"choices": [{"message": {"content": ""}}]}


async def agenerate_from_openai_chat_completion(
        messages_list: list[list[dict[str, str]]],
        engine: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        context_length: int,
        requests_per_minute: int = 300,
) -> list[str]:
    """Generate from OpenAI Chat Completion API.

    Args:
        messages_list: list of message list
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=engine,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for message in messages_list
    ]
    responses = await tqdm_asyncio.gather(*async_responses)

    return [x["choices"][0]["message"]["content"] for x in responses]


@retry_with_exponential_backoff
def generate_from_openai_chat_completion(
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        context_length: int,
        stop_token: str | None = None,
        num_outputs: int = 1,
        port=None,
        name=None,
) -> Union[str, list[str]]:
    # print('MODEL NAME',model,flush=True)
    start_time = time.time()
    if ('internvl' in model.lower() or "internvl2" in model.lower() or "vwa" in model.lower() ) and "aws" not in model.lower():
        client = OpenAI(
            api_key="YOUR_API_KEY", base_url=os.environ["OPENAI_API_BASE"]
        )
        models = client.models.list()
        if port:
            client = OpenAI(
                api_key="YOUR_API_KEY", base_url=port
            )
        try:
            response = client.chat.completions.create(
                model=model if not name else name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=num_outputs
            )

        except Exception as e:
            print(e)

    elif "gpt" in model.lower():
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=num_outputs,
            stop=stop_token
        )


    elif "llama3" in model.lower():
        # Used for running vllm models.

        print("WARNING: Using OPENAI_API_KEY=EMPTY")
        openai_api_key = "EMPTY"
        openai_api_base = os.environ["OPENAI_API_BASE"]
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        models = client.models.list()
        model = models.data[0].id
    elif "hjx" in model.lower():
        model = model[4:]
        server_url = 'http://127.0.0.1:20001'
        new_messages = []
        for message in messages:
            new_message = {}
            new_message['role'] = message['role']
            content = message['content']
            if type(content) == str:
                new_message['content'] = content
            else:
                new_content = []
                for con in content:
                    if con['type'] == 'image_url':
                        new_con = {}
                        new_con['type'] = 'image_url'
                        new_con['image_url'] = {'url': con['image_url']['url']}
                        new_content.append(new_con)
                        pass
                    else:
                        new_content.append(con)
                new_message['content'] = new_content
            new_messages.append(new_message)

        response = chat_with_openai(server_url, model, new_messages, temperature, max_tokens, top_p, num_outputs)
        print("Completion time:", time.time() - start_time)
        return response['answer']
    elif "aws" in model.lower():
        model = model[4:]
        port = os.environ["PORT"]
        response = chat_with_141(os.environ["LOCAL_API_SERVER"], model, messages,
                                    temperature, max_tokens, top_p, num_outputs, port=port)
        print("Completion time:", time.time() - start_time)
        print(response['answer'])
        return response['answer']

    elif "qwen" in model.lower():

        client = OpenAI(
            api_key="EMPTY",
            base_url=os.environ["OPENAI_API_BASE"],
        )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=num_outputs
            )
        except Exception as e:
            if "content management" in e.message:
                raise openai.error.InvalidRequestError(e.message)

    print("Completion time:", time.time() - start_time)
    if num_outputs > 1:
        answer: list[str] = [x.message.content for x in response.choices]
    else:
        answer: str = response.choices[0].message.content

    print("Answer:", answer, flush=True)

    return answer


@retry_with_exponential_backoff
# debug only
def fake_generate_from_openai_chat_completion(
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        context_length: int,
        stop_token: str | None = None,
) -> str:
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )

    answer = "Let's think step-by-step. This page shows a list of links and buttons. There is a search box with the label 'Search query'. I will click on the search box to type the query. So the action I will perform is \"click [60]\"."
    return answer


def chat_with_openai(server_url, model, messages, temperature=1.0, max_tokens=8192, top_p=1.0, num_outputs=1,
                     port="20000"):

    payload = {
        'model': model,
        'messages': messages,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_p': top_p,
        'n': num_outputs,
        'port': port
    }
    response = requests.post(server_url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return {'error': response.json().get('error', 'Unknown error occurred')}


def chat_with_141(server_url, model, messages, temperature=1.0, max_tokens=8192, top_p=1.0, num_outputs=1, port=2333):

    payload = {
        'model': model,
        'messages': messages,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_p': top_p,
        'n': num_outputs,
        "port": port
    }
    response = requests.post(server_url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return {'error': response.json().get('error', 'Unknown error occurred')}



def chat_with_ug(server_url, image, prompt):

    payload = {
        'image': image,
        'prompt': prompt,
    }
    response = requests.post(server_url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return {'error': response.json().get('error', 'Unknown error occurred')}

