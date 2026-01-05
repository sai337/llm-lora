from __future__ import annotations

def format_dolly(instruction: str, context: str | None, response: str | None = None) -> str:
    parts = []
    parts.append("### Instruction:\n" + instruction.strip())
    if context and str(context).strip():
        parts.append("\n### Context:\n" + str(context).strip())
    if response is not None:
        parts.append("\n### Response:\n" + str(response).strip())
    else:
        parts.append("\n### Response:\n")
    return "\n".join(parts) + "\n"

TEMPLATES = {"dolly": format_dolly}
