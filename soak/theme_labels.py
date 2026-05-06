"""Theme label generation for framework matrix column headers.

Generates short, unique labels for themes using an LLM call. Labels are
8-12 characters, distinct across the theme set, and suitable for matrix
column headers.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field
from struckdown.response_types import ResponseTypes
from struckdown.return_type_models import ResponseModel

logger = logging.getLogger(__name__)

LABEL_TEMPLATE = Path(__file__).parent / "templates" / "generate_theme_labels.sd"
BATCH_SIZE = 20


@ResponseTypes.register("ThemeLabel")
class ThemeLabel(ResponseModel):
    """A short label for a theme, keyed by theme hash."""

    theme_hash: str = Field(
        ..., description="The hash identifier of the theme being labelled."
    )
    short_label: str = Field(
        ...,
        max_length=25,
        description="A short, unique label for the theme (8-12 characters ideal, always <5 words).",
    )


async def generate_theme_labels(
    themes: List[Dict],
    model_name: Optional[str] = None,
    credentials: Optional[dict] = None,
) -> Dict[str, str]:
    """Generate short, unique labels for a list of themes.

    Args:
        themes: List of theme dicts with 'name', 'description', and a hash() method or
                'code_hash' key. Each theme needs a unique identifier.
        model_name: LLM model to use. Defaults to gpt-4.1-mini.
        credentials: LLM credentials dict.

    Returns:
        Dict mapping theme hash -> short label string.
    """
    from jinja2 import StrictUndefined, Template
    from struckdown import LLM, complete_async

    if not themes:
        return {}

    # compute hashes for themes
    theme_hashes = []
    for t in themes:
        if hasattr(t, "hash"):
            h = t.hash()
        elif isinstance(t, dict):
            from soak.models.base import compute_code_hash

            h = compute_code_hash(t.get("name", ""), t.get("description", ""))
        else:
            h = str(hash(str(t)))[:8]
        theme_hashes.append(h)

    llm = LLM(model_name=model_name or "gpt-4.1-mini")
    template_text = LABEL_TEMPLATE.read_text()
    template = Template(template_text, undefined=StrictUndefined)

    labels = {}
    prev_batch_text = ""

    # batch themes if >BATCH_SIZE
    for batch_start in range(0, len(themes), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(themes))
        batch_themes = themes[batch_start:batch_end]
        batch_hashes = theme_hashes[batch_start:batch_end]
        n = len(batch_themes)

        prompt = template.render(
            themes=batch_themes,
            theme_hashes=batch_hashes,
            N=n,
            prev_batch=bool(prev_batch_text),
            prev_batch_themes_and_labels=prev_batch_text,
        )

        result = await complete_async(
            multipart_prompt=prompt,
            model=llm,
            credentials=credentials,
        )

        # extract labels from result
        if hasattr(result, "outputs") and "theme_labels" in result.outputs:
            result_labels = result.outputs["theme_labels"]
            if isinstance(result_labels, list):
                for label_obj in result_labels:
                    if hasattr(label_obj, "theme_hash") and hasattr(
                        label_obj, "short_label"
                    ):
                        labels[label_obj.theme_hash] = label_obj.short_label
                    elif isinstance(label_obj, dict):
                        labels[label_obj["theme_hash"]] = label_obj["short_label"]

        # build prev_batch text for next batch
        for t, h in zip(batch_themes, batch_hashes):
            name = t.get("name", t.name) if isinstance(t, dict) else t.name
            label = labels.get(h, "?")
            prev_batch_text += f"[{h}] {name} -> {label}\n"

    logger.info(f"Generated {len(labels)} theme labels")
    return labels
