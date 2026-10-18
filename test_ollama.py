from ollama_client import OllamaClient


client = OllamaClient()

for chunk in client.generate_stream(
    "You're debugging a Node.js API where a request sometimes returns HTTP 500. Explain how you would investigate the problem step by step, including what logs, code paths, and tests you would inspect."
,temperature=0.7):
    print(chunk, end="", flush=True)

print()