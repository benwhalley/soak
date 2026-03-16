"""Context variable models for pipeline configuration.

Supports rich context variable definitions with descriptions and help text
for UI generation, while maintaining backward compatibility with simple values.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class ContextVariableType(str, Enum):
    """Type of context variable, controlling UI display location."""

    PRIMARY = "primary"  # shown prominently in results header
    SECONDARY = "secondary"  # shown in Data & Settings, same priority as cost/models
    CONFIGURATION = "configuration"  # shown in Data & Settings, smaller/less prominent


class ContextVariable(BaseModel):
    """Rich context variable definition with metadata for UI generation.

    Supports both simple string defaults and structured definitions with
    description and help_text for generating form fields in web UIs.

    Examples:
        Simple value (backward compatible):
            persona: Experienced qualitative researcher

        Rich definition:
            persona:
              default: Experienced qualitative researcher
              description: Analyst persona
              help_text: The perspective from which the analysis is conducted
              type: primary

        Type values control UI display location:
            - primary: shown prominently in results header next to TLDR
            - secondary: shown in Data & Settings tab, same priority as cost/models
            - configuration: shown in Data & Settings tab, smaller/less prominent
    """

    default: Any = ""
    description: str = ""
    help_text: str = ""
    required: bool = False
    input_type: str = "text"  # text, textarea, select, number
    type: Optional[str] = None  # primary, secondary, configuration

    # For select input_type
    options: List[str] = Field(default_factory=list)

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: Optional[str]) -> Optional[str]:
        """Validate type is one of the allowed values.

        Accepts 'config' as alias for 'configuration'.
        """
        if v is None:
            return None
        # Normalize 'config' to 'configuration'
        if v == "config":
            return "configuration"
        allowed = {"primary", "secondary", "configuration"}
        if v not in allowed:
            raise ValueError(f"type must be one of {allowed}, got '{v}'")
        return v

    @classmethod
    def from_value(cls, value: Any) -> "ContextVariable":
        """Create ContextVariable from a simple value or dict definition.

        Args:
            value: Either a simple value (str, int, etc.) or a dict with
                   default, description, help_text fields.

        Returns:
            ContextVariable instance
        """
        if isinstance(value, dict) and "default" in value:
            # Rich definition with default key
            return cls(
                default=value.get("default", ""),
                description=value.get("description", ""),
                help_text=value.get("help_text", ""),
                required=value.get("required", False),
                input_type=value.get("input_type", "text"),
                options=value.get("options", []),
                type=value.get("type"),
            )
        elif isinstance(value, cls):
            return value
        else:
            # Simple value -- just use as default
            return cls(default=value)

    @property
    def value(self) -> Any:
        """Return the default value (alias for backward compatibility)."""
        return self.default


def normalize_context(context: Dict[str, Any]) -> Dict[str, ContextVariable]:
    """Normalize context dict to ContextVariable instances.

    Converts a mix of simple values and rich definitions to a uniform
    dict of ContextVariable objects.

    Args:
        context: Dict that may contain simple values or ContextVariable dicts

    Returns:
        Dict mapping variable names to ContextVariable instances
    """
    return {name: ContextVariable.from_value(value) for name, value in context.items()}


def get_context_defaults(context: Dict[str, Any]) -> Dict[str, Any]:
    """Extract default values from context (simple or rich format).

    Args:
        context: Dict that may contain simple values or ContextVariable dicts

    Returns:
        Dict mapping variable names to their default values
    """
    result = {}
    for name, value in context.items():
        if isinstance(value, ContextVariable):
            result[name] = value.default
        elif isinstance(value, dict) and "default" in value:
            result[name] = value["default"]
        else:
            result[name] = value
    return result
