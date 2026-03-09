import asyncio
import os
import time

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()


class RateLimitedOpenAIClient(OpenAIChatCompletionClient):
    """带有速率限制控制和重试机制的 OpenAI 客户端包装器"""
    
    def __init__(self, *args, delay_seconds: float = 3.0, max_retries: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self._delay_seconds = delay_seconds
        self._max_retries = max_retries
        self._last_call_time = 0
    
    async def _wait_if_needed(self):
        """在每次调用前等待，确保不超过速率限制"""
        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        
        if time_since_last_call < self._delay_seconds:
            wait_time = self._delay_seconds - time_since_last_call
            print(f"[速率控制] 等待 {wait_time:.1f} 秒...")
            await asyncio.sleep(wait_time)
        
        self._last_call_time = time.time()
    
    async def create(self, messages, *args, **kwargs):
        for attempt in range(self._max_retries):
            try:
                await self._wait_if_needed()
                return await super().create(messages, *args, **kwargs)
            except Exception as e:
                if "速率限制" in str(e) or "RateLimitError" in str(type(e)):
                    wait_time = (attempt + 1) * 5  # 指数退避: 5s, 10s, 15s
                    print(f"[速率限制] 第 {attempt + 1} 次重试，等待 {wait_time} 秒...")
                    await asyncio.sleep(wait_time)
                    if attempt == self._max_retries - 1:
                        raise
                else:
                    raise
    
    async def create_stream(self, messages, *args, **kwargs):
        for attempt in range(self._max_retries):
            try:
                await self._wait_if_needed()
                async for chunk in super().create_stream(messages, *args, **kwargs):
                    yield chunk
                return
            except Exception as e:
                if "速率限制" in str(e) or "RateLimitError" in str(type(e)):
                    wait_time = (attempt + 1) * 5
                    print(f"[速率限制] 第 {attempt + 1} 次重试，等待 {wait_time} 秒...")
                    await asyncio.sleep(wait_time)
                    if attempt == self._max_retries - 1:
                        raise
                else:
                    raise


async def main():
    # 使用带速率限制的客户端，每次调用间隔 3 秒
    model_client = RateLimitedOpenAIClient(
        model=os.getenv("LLM_MODEL_ID", "gpt-4o"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        delay_seconds=3.0,  # 每次 API 调用间隔 3 秒
        max_retries=3,  # 遇到速率限制时最多重试 3 次
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
        },
    )

    planner_agent = AssistantAgent(
        "planner_agent",
        model_client=model_client,
        description="A helpful assistant that can plan trips.",
        system_message="You are a helpful assistant that can suggest a travel plan for a user based on their request.",
    )

    local_agent = AssistantAgent(
        "local_agent",
        model_client=model_client,
        description="A local assistant that can suggest local activities or places to visit.",
        system_message="You are a helpful assistant that can suggest authentic and interesting local activities or places to visit for a user and can utilize any context information provided.",
    )

    language_agent = AssistantAgent(
        "language_agent",
        model_client=model_client,
        description="A helpful assistant that can provide language tips for a given destination.",
        system_message="You are a helpful assistant that can review travel plans, providing feedback on important/critical tips about how best to address language or communication challenges for the given destination. If the plan already includes language tips, you can mention that the plan is satisfactory, with rationale.",
    )

    travel_summary_agent = AssistantAgent(
        "travel_summary_agent",
        model_client=model_client,
        description="A helpful assistant that can summarize the travel plan.",
        system_message="You are a helpful assistant that can take in all of the suggestions and advice from the other agents and provide a detailed final travel plan. You must ensure that the final plan is integrated and complete. YOUR FINAL RESPONSE MUST BE THE COMPLETE PLAN. When the plan is complete and all perspectives are integrated, you can respond with TERMINATE.",
    )

    termination = TextMentionTermination("TERMINATE")
    group_chat = RoundRobinGroupChat(
        [planner_agent, local_agent, language_agent, travel_summary_agent], termination_condition=termination
    )
    await Console(group_chat.run_stream(task="Plan a 3 day trip to Nepal."))

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
