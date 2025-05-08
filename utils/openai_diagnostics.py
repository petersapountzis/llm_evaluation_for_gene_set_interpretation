import os
from openai import OpenAI
import time
from datetime import datetime
import json
from dotenv import load_dotenv
load_dotenv()


class OpenAIDiagnostics:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)

    def run_diagnostics(self):
        print("\n=== OpenAI API Diagnostics ===")
        print(f"Time: {datetime.now()}")

        # Test 1: API Key Check
        self._test_api_key()

        # Test 2: Basic API Call
        self._test_basic_api_call()

        # Test 3: Model Access
        self._test_model_access()

        # Test 4: Rate Limit Test
        self._test_rate_limits()

    def _test_api_key(self):
        print("\n1. API Key Check:")
        if not self.api_key:
            print("❌ No API key found")
            return

        print(f"✓ API key found (starts with: {self.api_key[:8]}...)")
        print(f"✓ API key length: {len(self.api_key)} characters")

    def _test_basic_api_call(self):
        print("\n2. Basic API Call Test:")
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            end_time = time.time()

            print(f"✓ API call successful")
            print(f"✓ Response time: {(end_time - start_time):.2f} seconds")
            print(f"✓ Tokens used: {response.usage.total_tokens}")

        except Exception as e:
            print(f"❌ API call failed: {str(e)}")
            self._analyze_error(e)

    def _test_model_access(self):
        print("\n3. Model Access Test:")
        models = [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo-preview"
        ]

        for model in models:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=5
                )
                print(f"✓ {model}: Available")
            except Exception as e:
                print(f"❌ {model}: {str(e)}")

    def _test_rate_limits(self):
        print("\n4. Rate Limit Test:")
        try:
            # Make 3 quick requests to test rate limiting
            for i in range(3):
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": f"Test {i}"}],
                    max_tokens=5
                )
                print(f"✓ Request {i + 1}/3 successful")
                time.sleep(0.5)  # Small delay between requests

        except Exception as e:
            print(f"❌ Rate limit test failed on request {i + 1}: {str(e)}")
            self._analyze_error(e)

    def _analyze_error(self, error):
        print("\nError Analysis:")
        error_str = str(error)

        if "insufficient_quota" in error_str:
            print("🔍 Issue: Your account has reached its quota or has billing issues")
            print("   Solution: Check your billing status and payment method at platform.openai.com")

        elif "rate_limit" in error_str:
            print("🔍 Issue: You're making too many requests too quickly")
            print("   Solution: Implement rate limiting in your code")

        elif "invalid_api_key" in error_str:
            print("🔍 Issue: Your API key is invalid or malformed")
            print("   Solution: Double-check your API key in your .env file")

        else:
            print(f"🔍 Unrecognized error: {error_str}")


if __name__ == "__main__":
    diagnostics = OpenAIDiagnostics()
    diagnostics.run_diagnostics()