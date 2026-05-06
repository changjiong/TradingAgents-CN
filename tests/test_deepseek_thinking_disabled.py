import unittest
import sys
import types
from unittest.mock import patch

if "langchain_openai" not in sys.modules:
    fake_module = types.ModuleType("langchain_openai")

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module.ChatOpenAI = FakeChatOpenAI
    sys.modules["langchain_openai"] = fake_module

from tradingagents.llm_clients.openai_client import OpenAIClient


class DeepSeekThinkingDisabledTests(unittest.TestCase):
    def test_deepseek_client_disables_thinking_mode(self):
        with patch("tradingagents.llm_clients.openai_client.NormalizedChatOpenAI") as chat:
            OpenAIClient(
                "deepseek-v4-pro",
                provider="deepseek",
                api_key="sk-test",
            ).get_llm()

        kwargs = chat.call_args.kwargs
        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "disabled"}})

    def test_non_deepseek_client_does_not_set_thinking_body(self):
        with patch("tradingagents.llm_clients.openai_client.NormalizedChatOpenAI") as chat:
            OpenAIClient(
                "qwen-turbo",
                provider="qwen",
                api_key="sk-test",
            ).get_llm()

        kwargs = chat.call_args.kwargs
        self.assertNotIn("extra_body", kwargs)


if __name__ == "__main__":
    unittest.main()
