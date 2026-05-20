#!/usr/bin/env python3
"""
CLI demo for HealthCopilot.run_end_to_end — prints steps and writes e2e_result.json.

Functions:
    main — parse args (--user, --pdf, --feedback, --locale, --send-whatsapp), run copilot

Usage (from repo root):
    PYTHONPATH=. python scripts/run_e2e_demo.py
    PYTHONPATH=. python scripts/run_e2e_demo.py --user user_demo_001 --pdf sample_report_for ocr_demo.pdf
    PYTHONPATH=. python scripts/run_e2e_demo.py --locale hi --feedback "Replace breakfast"
"""

from __future__ import annotations

import argparse
import json
import sys

from orchestrator.copilot import get_copilot


def main() -> int:
    parser = argparse.ArgumentParser(description="Health Copilot E2E demo")
    parser.add_argument("--user", default="user_demo_001", help="User ID")
    parser.add_argument("--pdf", default=None, help="Optional lab PDF path")
    parser.add_argument("--feedback", default="No paneer tonight")
    parser.add_argument("--locale", default="en", choices=["en", "hi"])
    parser.add_argument("--send-whatsapp", action="store_true")
    parser.add_argument("--phone", default=None)
    args = parser.parse_args()

    copilot = get_copilot()
    result = copilot.run_end_to_end(
        args.user,
        pdf_path=args.pdf,
        feedback_text=args.feedback,
        locale=args.locale,
        send_whatsapp=args.send_whatsapp,
        phone=args.phone,
    )

    print("\n" + "=" * 60)
    print(f"E2E Copilot | user={args.user} | success={result.success}")
    print("=" * 60)
    for step in result.steps:
        icon = {"done": "✅", "error": "❌", "skipped": "⏭️", "running": "🔄"}.get(
            step.status, "•"
        )
        print(f"  {icon} {step.to_dict()['label']}: {step.message}")
    if result.errors:
        print("\nErrors:")
        for err in result.errors:
            print(f"  - {err}")
    print("\nWhatsApp preview (excerpt):")
    print((result.whatsapp_preview or "")[:500])
    print("\nFull JSON written to e2e_result.json")
    with open("e2e_result.json", "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
