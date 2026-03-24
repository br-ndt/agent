#!/usr/bin/env python3
"""Quick connectivity test — run after setting up .env.

Usage: uv run python test_connection.py
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()


async def test_claude_cli():
    import shutil
    if not shutil.which("claude"):
        print("  ⏭  claude CLI not installed")
        print("     Install: npm install -g @anthropic-ai/claude-code")
        print("     Then:    claude login")
        return

    from agent.providers.claude_cli import ClaudeCLIProvider

    p = ClaudeCLIProvider(timeout=60)
    r = await p.complete(
        messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        model="sonnet",
        max_tokens=50,
    )
    print(f"  ✓  Claude CLI: {r.content!r} (model={r.model}, usage={r.usage})")
    print(f"     (Using your Pro/Max subscription — no API cost)")


async def test_anthropic():
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        print("  ⏭  ANTHROPIC_API_KEY not set, skipping")
        return

    from agent.providers.anthropic import AnthropicProvider

    p = AnthropicProvider(key)
    r = await p.complete(
        messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        model="claude-sonnet-4-6",
        max_tokens=50,
    )
    print(f"  ✓  Anthropic: {r.content!r} (model={r.model}, tokens={r.usage})")


async def test_openai():
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        print("  ⏭  OPENAI_API_KEY not set, skipping")
        return

    from agent.providers.openai import OpenAIProvider

    p = OpenAIProvider(key)
    r = await p.complete(
        messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        model="gpt-4o-mini",
        max_tokens=50,
    )
    print(f"  ✓  OpenAI: {r.content!r} (model={r.model}, tokens={r.usage})")


async def test_google():
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key:
        print("  ⏭  GOOGLE_API_KEY not set, skipping")
        return

    from agent.providers.google import GoogleProvider

    p = GoogleProvider(key)
    r = await p.complete(
        messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        model="gemini-2.5-flash",
        max_tokens=50,
    )
    print(f"  ✓  Google: {r.content!r} (model={r.model}, tokens={r.usage})")


async def main():
    print("Testing LLM provider connectivity...\n")

    for name, test_fn in [
        ("Claude CLI", test_claude_cli),
        ("Anthropic API", test_anthropic),
        ("OpenAI", test_openai),
        ("Google", test_google),
    ]:
        try:
            await test_fn()
        except Exception as e:
            print(f"  ✗  {name}: {type(e).__name__}: {e}")

    print("\nDone. Set any missing API keys in .env and re-run.")


if __name__ == "__main__":
    asyncio.run(main())
