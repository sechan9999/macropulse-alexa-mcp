"""
scripts/eval_briefing_providers.py
─────────────────────────────────────────────────────────────────
Head-to-head eval of get_macro_briefing's two LLM providers (Gemini vs.
Amazon Bedrock) on the same live macro data and the same prompt, so you
can actually compare briefing quality rather than assume one is better.

Not runnable in CI or in this repo's sandbox — needs a real GEMINI_API_KEY
and real AWS credentials with Bedrock access. Run it yourself once both
are configured:

    python scripts/eval_briefing_providers.py
    python scripts/eval_briefing_providers.py --analysis-type "Risk Assessment"
    python scripts/eval_briefing_providers.py --bedrock-model anthropic.claude-3-5-sonnet-20241022-v2:0

Writes a side-by-side markdown report to stdout (redirect to a file if
you want to keep it) with both briefings plus latency for each call.
This is a comparison aid, not an automated judge — read both and decide
which you'd ship.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `src.*`

from src.macro_briefing import ANALYSIS_TYPES, generate_briefing


def _run(provider: str, analysis_type: str, custom_question: str, model_id: str | None) -> dict:
    start = time.monotonic()
    result = generate_briefing(
        analysis_type=analysis_type,
        custom_question=custom_question,
        provider=provider,
        model_id=model_id,
    )
    result["_elapsed_s"] = round(time.monotonic() - start, 1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-type", default="Full Macro Briefing", choices=ANALYSIS_TYPES)
    parser.add_argument("--custom-question", default="",
                         help="Required if --analysis-type='Custom Question'")
    parser.add_argument("--bedrock-model", default=None,
                         help="Override the default Bedrock model (amazon.nova-pro-v1:0)")
    parser.add_argument("--gemini-model", default=None,
                         help="Override the default Gemini model (gemini-3.6-flash)")
    args = parser.parse_args()

    print(f"# Briefing eval — {args.analysis_type}\n")

    gemini = _run("gemini", args.analysis_type, args.custom_question, args.gemini_model)
    bedrock = _run("bedrock", args.analysis_type, args.custom_question, args.bedrock_model)

    for label, result in (("Gemini", gemini), ("Bedrock", bedrock)):
        print(f"## {label}  ({result.get('model', '?')}, {result['_elapsed_s']}s)\n")
        if "error" in result:
            print(f"**ERROR:** {result['error']}\n")
        else:
            print(result["text"])
            print()
        print("---\n")

    if "error" not in gemini and "error" not in bedrock:
        print("## Compare\n")
        print(f"- Gemini: {gemini['_elapsed_s']}s")
        print(f"- Bedrock: {bedrock['_elapsed_s']}s")
        print("\nRead both briefings above and judge on: grounding in the actual "
              "numbers, actionability of the recommendations, and whether either "
              "one hallucinated a figure not present in the context.")
        return 0

    print("One or both providers failed — fix the error above before comparing quality.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
