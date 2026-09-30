"""Read the same MGI guideline for the policy prompt and frozen reward judge."""
import json
from pathlib import Path
from string import Formatter


def load_guideline(path, default):
    if not path:
        return default
    source = Path(path).expanduser()
    text = source.read_text(encoding="utf-8")
    template = json.loads(text)["evolve_template"] if source.suffix == ".json" else text
    fields = {field for _, field, _, _ in Formatter().parse(template) if field is not None}
    if fields != {"memory", "chunk"}:
        raise ValueError("Memory guideline must use exactly {memory} and {chunk}; escape literal braces")
    template.format(memory="", chunk="")
    return template
