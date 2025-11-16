import os
import asyncio
import random
import weave
from typing import Optional

import time
from litellm import acompletion

from dotenv import load_dotenv
load_dotenv()

from openai import RateLimitError

# Handle OPENROUTER_LLM_KEY -> OPENROUTER_API_KEY mapping
if os.getenv("OPENROUTER_LLM_KEY") and not os.getenv("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_LLM_KEY")


MODEL_MAP = {
    "gpt-4o-mini": "gpt-4o-mini",
    "claude-3-5-sonnet-20240620": "claude-3-5-sonnet-20240620",
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4-turbo": "gpt-4-turbo",
    "o1-preview": "o1-preview",
    "o1-mini": "o1-mini",
    "claude-3-opus-20240229": "claude-3-opus-20240229",
    "command-r-plus": "command-r-plus-08-2024",
    "gemini-1.5-pro": "gemini/gemini-1.5-pro",
    "llama3-405b-instruct": "fireworks_ai/accounts/fireworks/models/llama-v3p1-405b-instruct",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "gemini-1.5-pro-002": "gemini/gemini-1.5-pro-002",
    "mistral-large": "mistral/mistral-large-2407",
    "grok-2": "openrouter/x-ai/grok-2"
}

EXPONENTIAL_BASE = 2    


class MajorityVoteModel(weave.Model):
    model: weave.Model
    num_responses: int = 3
    
    @weave.op()
    async def predict(self, prompt: str):
        tasks = [self.model.predict(prompt) for _ in range(self.num_responses)]
        return await asyncio.gather(*tasks)


class LiteLLMModel(weave.Model):
    model_name: str
    system_prompt: Optional[str] = None
    temp: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    max_retries: int = 3
    
    def __init__(self, **data):
        super().__init__(**data)
        # Add any additional initialization logic here
        # Allow openrouter models that aren't in MODEL_MAP
        if self.model_name not in MODEL_MAP and not self.model_name.startswith("openrouter/"):
            raise ValueError(f"Invalid model name: {self.model_name}")

        if "o1" in self.model_name:
            self.temp = None
            self.top_p = None
            self.max_tokens = None

    
    @weave.op()
    async def predict(self, prompt: str):
        delay = 2

        for i in range(self.max_retries):
            try:
                messages = []
                if self.system_prompt is not None:
                    messages.append({
                        "role": "system",
                        "content": self.system_prompt
                    })
                messages.append({
                    "role": "user",
                    "content": prompt
                })

                # Use MODEL_MAP if available, otherwise use model_name directly
                model_to_use = MODEL_MAP.get(self.model_name, self.model_name)

                # For OpenRouter stealth models (no provider in path like openrouter/sherlock-dash-alpha),
                # LiteLLM needs the format openrouter/openrouter/model-name
                if (self.model_name.startswith("openrouter/") and
                    "/" not in self.model_name[11:]):  # No provider after openrouter/
                    # Double the openrouter prefix for LiteLLM
                    model_to_use = "openrouter/" + self.model_name

                response = await acompletion(
                    model=model_to_use,
                    messages=messages,
                    temperature=self.temp,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p
                )

                if response.choices[0].message.content is not None:
                    return response.choices[0].message.content
                else:
                    print(response)
                    raise Exception("No content in response")
            except RateLimitError as e:
                delay *= EXPONENTIAL_BASE * (1 + random.random())
                print(
                    f"RateLimitError, retrying after {round(delay, 2)} seconds, {i+1}-th retry...", e
                )
                await asyncio.sleep(delay)
                continue
            except Exception as e:
                print(f"Error in retry {i+1}, retrying...", e)
                continue

        raise Exception("Failed to get response after max retries")
