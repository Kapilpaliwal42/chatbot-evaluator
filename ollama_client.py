import json
import requests


class OllamaClient:
    def __init__(
        self,
        base_url="http://127.0.0.1:11434",
        model="qwen2.5-coder:7b",
        timeout=120,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def generate(
    self,
    prompt,
    temperature=0.7,
    num_predict=None,
    num_ctx=None,
    format=None,
):
        url = f"{self.base_url}/api/generate"

        payload = {
    "model": self.model,
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": temperature,
    },
    
}

        if num_predict is not None:
            payload["options"]["num_predict"] = num_predict

        if num_ctx is not None:
            payload["options"]["num_ctx"] = num_ctx

        if format is not None:
            payload["format"] = format

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Could not connect to Ollama. "
                "Make sure Ollama is running."
            )

        except requests.exceptions.Timeout:
            raise RuntimeError(
                "The request to Ollama timed out."
            )

        data = response.json()

        if "response" not in data:
            raise RuntimeError(
                "Ollama returned an unexpected response."
            )

        return data["response"]
    def generate_stream(
        self,
        prompt,
        temperature=0.7,
        num_predict=None,
        num_ctx=None,
        format=None,
    ):
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
            },
        }

        if num_predict is not None:
            payload["options"]["num_predict"] = num_predict

        if num_ctx is not None:
            payload["options"]["num_ctx"] = num_ctx

        if format is not None:
            payload["format"] = format

        try:
            response = requests.post(
                url,
                json=payload,
                stream=True,
                timeout=self.timeout,
            )

            response.raise_for_status()

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Could not connect to Ollama. "
                "Make sure Ollama is running."
            )

        except requests.exceptions.Timeout:
            raise RuntimeError(
                "The request to Ollama timed out."
            )

        for line in response.iter_lines():
            if not line:
                continue

            data = json.loads(line.decode("utf-8"))

            if "error" in data:
                raise RuntimeError(data["error"])

            chunk = data.get("response", "")

            if chunk:
                yield chunk

            if data.get("done"):
                break

