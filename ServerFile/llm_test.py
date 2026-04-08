import argparse
import json
import sys
import time
from dataclasses import dataclass

import requests


@dataclass
class ChatMessage:
    role: str
    content: str


def load_description():
    from Description import Description

    return Description()


def call_ark_chat(api_key, model, system_prompt, prompt, temperature=0.2, timeout_s=20):
    """Directly call VolcEngine Ark chat/completions endpoint."""
    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "stream": False,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"], data


def call_project_wrapper(system_prompt, prompt):
    """Call the existing project wrapper in volcEngineLLM.py."""
    from volcEngineLLM import VolcEngineFakeHFModel

    model = VolcEngineFakeHFModel()
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=prompt),
    ]
    result = model.generate(messages)
    return result.content


def main():
    parser = argparse.ArgumentParser(description="LLM connectivity and response test.")
    parser.add_argument(
        "--mode",
        choices=["direct", "wrapper"],
        default="wrapper",
        help="direct: call Ark API directly; wrapper: call project volcEngineLLM wrapper.",
    )
    parser.add_argument(
        "--prompt",
        default="搜索红色气球",
        help="Prompt for the LLM.",
    )
    parser.add_argument(
        "--dump-description",
        action="store_true",
        help="Print the current system prompt only, then exit.",
    )
    parser.add_argument("--model", default="deepseek-v3-250324", help="Model ID for direct mode.")
    parser.add_argument("--api-key", default="", help="API key for direct mode.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--timeout", type=int, default=20, help="Request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=1, help="Retry count when request fails.")
    args = parser.parse_args()

    description = load_description()
    system_prompt = description.Prompt_dit["Prompt_smol"]

    if args.dump_description:
        print(system_prompt)
        return 0

    last_err = None
    for i in range(1, args.retries + 1):
        try:
            t0 = time.time()
            if args.mode == "direct":
                if not args.api_key:
                    raise ValueError("--api-key is required in direct mode")
                text, raw = call_ark_chat(
                    api_key=args.api_key,
                    model=args.model,
                    system_prompt=system_prompt,
                    prompt=args.prompt,
                    temperature=args.temperature,
                    timeout_s=args.timeout,
                )
                print(text)
                usage = raw.get("usage", {})
                if usage:
                    pass
            else:
                text = call_project_wrapper(system_prompt, args.prompt)
                print(text)

            return 0
        except Exception as e:
            last_err = e
            print(f"[LLM_TEST] attempt {i}/{args.retries} failed: {e}", file=sys.stderr)
            if i < args.retries:
                time.sleep(1.0)

    print(f"[LLM_TEST] FAILED: {last_err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
