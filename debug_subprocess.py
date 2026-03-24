"""Run on VPS: uv run python debug_subprocess.py"""
 
import asyncio
import json
 
 
async def main():
    cmd = ["claude", "-p", "--output-format", "json", "--model", "sonnet"]
    prompt = "Say hello and nothing else."
 
    print(f"Command: {' '.join(cmd)}")
    print(f"Piping via stdin: {prompt!r}")
    print()
 
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
 
    stdout_bytes, stderr_bytes = await asyncio.wait_for(
        proc.communicate(input=prompt.encode("utf-8")),
        timeout=60,
    )
 
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
 
    print(f"Return code: {proc.returncode}")
    print(f"Stdout length: {len(stdout)}")
    print(f"Stderr length: {len(stderr)}")
    print()
    print(f"=== STDOUT ===")
    print(repr(stdout[:2000]))
    print()
    print(f"=== STDERR ===")
    print(repr(stderr[:2000]))
 
    if stdout.strip():
        try:
            parsed = json.loads(stdout.strip())
            print()
            print("=== PARSED JSON KEYS ===")
            print(list(parsed.keys()))
            if "result" in parsed:
                print(f"result: {parsed['result']!r}")
        except json.JSONDecodeError as e:
            print(f"\nJSON parse failed: {e}")
            print("First 500 chars:", stdout[:500])
 
 
asyncio.run(main())