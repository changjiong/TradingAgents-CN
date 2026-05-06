import unittest
from unittest.mock import Mock, patch

from app.services.config_service import ConfigService


class DeepSeekV4ResponseTests(unittest.TestCase):
    def test_deepseek_test_accepts_reasoning_content_when_content_empty(self):
        service = ConfigService()
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": "这是模型的思考内容。",
                    }
                }
            ]
        }

        with patch("requests.post", return_value=response):
            result = service._test_deepseek_api(
                "sk-test",
                "deepseek deepseek-v4-pro",
                "deepseek-v4-pro",
            )

        self.assertTrue(result["success"])


if __name__ == "__main__":
    unittest.main()
