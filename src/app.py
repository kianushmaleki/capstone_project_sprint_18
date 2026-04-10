import os
from dotenv import load_dotenv
import anthropic

load_dotenv()


def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def chat(prompt: str, model: str = "claude-sonnet-4-6") -> str:
    client = get_client()
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def main():
    print("LLM Interface — type 'quit' to exit")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue
        response = chat(user_input)
        print(f"\nAssistant: {response}")


if __name__ == "__main__":
    main()
