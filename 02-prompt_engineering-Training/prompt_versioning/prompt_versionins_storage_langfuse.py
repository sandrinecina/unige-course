# -*- coding: utf-8 -*-
"""
Minimal demo: store prompts in Langfuse, retrieve by 'production' label, run & log.

Env:
  OPENAI_API_KEY=...
  LANGFUSE_PUBLIC_KEY=...
  LANGFUSE_SECRET_KEY=...
  # optional: LANGFUSE_HOST=https://cloud.langfuse.com
"""

import os
from langfuse import get_client
from langfuse.openai import openai as openai_client
from langfuse.api.resources.commons.errors.not_found_error import NotFoundError

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

PROMPTS = {
    "v1_simple_instruction": "Answer concisely:\n\n{{text}}",
    "v2_expert_tone": "You are an expert assistant. Be thorough and helpful.\n\nUser: {{text}}\n\nAssistant:",
}

QUESTION = "List three best practices for secure API design."

def main():
    # Clients
    lf = get_client()                 # uses LANGFUSE_PUBLIC_KEY/SECRET_KEY(/HOST)
    oai = openai_client.OpenAI()      # uses OPENAI_API_KEY

    # 1) Seed/version prompts and label them as 'production'
    for name, template in PROMPTS.items():
        lf.create_prompt(
            name=name,
            type="text",
            prompt=template,
            labels=["production"],    # <-- this makes get_prompt(..., label="production") work
            # optional:
            # commit_message="seed/update",
            # config={"model": MODEL, "temperature": 0.2},
        )
    print("[Langfuse] prompts created/versioned & labeled 'production'")

    # 2) Retrieve → compile → run (auto-logged) for each version
    for name in PROMPTS:
        try:
            p = lf.get_prompt(name, label="production")  # fetch the production-labeled version
        except NotFoundError:
            # Fallback to latest if not labeled yet
            p = lf.get_prompt(name)

        compiled = p.compile(text=QUESTION)  # fill {{text}}

        resp = oai.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": compiled}],
            langfuse_prompt=p,               # link this run to the exact prompt version in Langfuse
        )

        out = resp.choices[0].message.content.strip()
        usage = getattr(resp, "usage", None)
        total = getattr(usage, "total_tokens", 0) if usage else 0

        print(f"\n=== {name} (production) ===")
        print(f"Tokens: {total}")
        print(out)

if __name__ == "__main__":
    main()
