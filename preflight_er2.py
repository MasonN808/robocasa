#!/usr/bin/env python
"""Checks everything ER 2 needs, in order, before any eval spend.

Vertex is not an option for this model: gemini-robotics-er-2-preview and
gemini-robotics-er-1.5-preview both 404 on every region tested with valid
ADC, so the Gemini Developer API and a working API key are the only route.

  python preflight_er2.py [--model gemini-robotics-er-2-preview]

Exit 0 means the model is reachable AND supports the generate_content +
function-calling contract live_sim_eval/eval_standalone depend on. Any other
exit prints the specific next action rather than a stack trace.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ENV_FILE = Path("/work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env")


def load_key() -> str | None:
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    if not ENV_FILE.exists():
        return None
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            return v.strip().strip("\"'")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-robotics-er-2-preview")
    args = ap.parse_args()

    key = load_key()
    if not key:
        print(f"FAIL  no GEMINI_API_KEY found in env or {ENV_FILE}")
        return 2
    print(f"ok    key found (len={len(key)}, prefix_ok={key.startswith('AIza')})")

    from google import genai

    client = genai.Client(api_key=key)

    # 1. Does the key authenticate at all?
    try:
        models = list(client.models.list())
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  key rejected: {type(exc).__name__}: {str(exc)[:160]}")
        print("      -> regenerate at aistudio.google.com/apikey and confirm the")
        print("         Generative Language API is enabled on that project.")
        return 3
    print(f"ok    key authenticates ({len(models)} models visible)")

    # 2. Is this specific model exposed to this key?
    names = {m.name.split("/")[-1]: m for m in models}
    hit = names.get(args.model)
    if hit is None:
        robotics = sorted(n for n in names if "robot" in n or "-er-" in n)
        print(f"FAIL  {args.model} not visible to this key")
        print(f"      robotics-ish models this key CAN see: {robotics or 'none'}")
        print("      -> the model is access-gated; request access for this project.")
        return 4
    print(f"ok    {args.model} is visible")

    # 3. The harness calls generate_content with function declarations. A model
    #    exposed only through the Live API (bidiGenerateContent) cannot be driven
    #    by that path, and would need a separate streaming policy.
    methods = list(getattr(hit, "supported_actions", None) or [])
    if methods and "generateContent" not in methods:
        print(f"FAIL  supported methods = {methods}")
        print("      -> no generateContent; the harness would need a Live-API policy.")
        return 5
    print(f"ok    supports generateContent (methods={methods or 'unreported'})")

    # 4. Function calling is the whole contract -- a model that ignores tools is
    #    useless here even if it responds. Verify an actual forced call.
    from google.genai import types

    decl = types.FunctionDeclaration(
        name="navigate_to_fixture",
        description="Move an agent to a fixture.",
        parameters={
            "type": "OBJECT",
            "properties": {"fixture": {"type": "STRING"}},
            "required": ["fixture"],
        },
    )
    try:
        resp = client.models.generate_content(
            model=args.model,
            contents="Go to the sink.",
            config=types.GenerateContentConfig(
                tools=[types.Tool(function_declarations=[decl])],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="ANY")
                ),
                temperature=0.0,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  function-calling probe: {type(exc).__name__}: {str(exc)[:160]}")
        return 6

    calls = [
        p.function_call
        for c in (resp.candidates or [])
        for p in (getattr(c.content, "parts", None) or [])
        if getattr(p, "function_call", None)
    ]
    if not calls:
        print("FAIL  model returned no function call under mode=ANY")
        print(f"      text was: {(resp.text or '')[:120]!r}")
        return 7
    print(f"ok    function calling works -> {calls[0].name}({dict(calls[0].args or {})})")
    print("\nPASS  ready to launch ER 2 evals.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
