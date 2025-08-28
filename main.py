from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled
from dotenv import load_dotenv, find_dotenv
from rich import print
import os

# Disable tracing
set_tracing_disabled(disabled=True)

# Load .env file
load_dotenv(find_dotenv(raise_error_if_not_found=True))

# ✅ Get the GEMINI_API_KEY from environment
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("❌ GEMINI_API_KEY not found. Make sure it's set in your .env file.")

# External client
externel_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Chat model
llm_model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=externel_client
)

# ------------------------------
# System prompt with fixed Q/As
# ------------------------------
faq_instructions = """
You are a helpful FAQ bot. 

You must always reply with the exact predefined answers below if the user asks one of these questions:

- "what is your name?" → I am FAQBot 🤖, your helpful assistant!
- "what can you do?" → I can answer predefined questions quickly and clearly.
- "how are you?" → I’m just code, but I’m doing great! 🚀
- "who created you?" → I was created by a developer using the OpenAI Agent SDK + Gemini Flash 2.5.
- "what is chainlit?" → Chainlit is a framework to build beautiful UIs for LLM-powered apps.

If the question is not in this list, politely say: "Sorry, I don’t know the answer to that."
"""

# Agent
faq_agent = Agent(
    name="faq_agent",
    instructions=faq_instructions,
    model=llm_model,
)

# Run
result = Runner.run_sync(
    faq_agent,
    "what is your name?",
)

print(result.final_output)
