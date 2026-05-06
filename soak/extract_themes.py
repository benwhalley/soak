"""Extract structured themes from free text via LLM.

Used for creating analytical frameworks from papers, notes, or other text.
"""

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).parent / "templates" / "theme_extraction.sd"


async def extract_themes(
    text: str,
    prompt: Optional[str] = None,
    model: Optional[str] = None,
    credentials: Optional[dict] = None,
) -> List[dict]:
    """Extract structured themes from free text via a single LLM call.

    Args:
        text: Source text (paper excerpt, notes, framework description, etc.)
        prompt: Optional guidance for the extraction.
        model: Model to use. Falls back to gpt-4.1-mini.
        credentials: LLM credentials dict.

    Returns:
        List of dicts with 'name' and 'description' keys.
    """
    from jinja2 import StrictUndefined, Template
    from struckdown import LLM, complete_async

    llm = LLM(model_name=model or "gpt-4.1-mini")
    template_text = TEMPLATE_PATH.read_text()
    template = Template(template_text, undefined=StrictUndefined)

    rendered = template.render(input_text=text, prompt=prompt or "")

    result = await complete_async(
        multipart_prompt=rendered,
        model=llm,
        credentials=credentials,
    )

    themes = []
    if hasattr(result, "outputs") and "themes" in result.outputs:
        raw_themes = result.outputs["themes"]
        if isinstance(raw_themes, list):
            for t in raw_themes:
                if hasattr(t, "name") and hasattr(t, "description"):
                    themes.append({"name": t.name, "description": t.description})
                elif isinstance(t, dict):
                    themes.append(
                        {"name": t.get("name", ""), "description": t.get("description", "")}
                    )

    logger.info(f"Extracted {len(themes)} themes from text")
    return themes
