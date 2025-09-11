# utils/naming.py

import re

from django.db.models.query import QuerySet


def get_available_name(
    base_name: str,
    queryset: QuerySet,
    name_field: str = "name",
    suffix: str = " (copy)",
) -> str:
    """
    Generate a unique name for a copied object.

    Removes any existing ' (copy)' or ' (copy N)' suffix and appends a new one
    that does not clash with existing names in the queryset.

    Args:
        base_name (str): The starting name to copy.
        queryset (QuerySet): A queryset of the model to check uniqueness.
        name_field (str): The name of the field to check uniqueness on.
        suffix (str): The suffix to append (default: ' (copy)').

    Returns:
        str: A unique name.
    """
    # Remove trailing ' (copy)' or ' (copy N)'
    base = re.sub(rf"{re.escape(suffix)}(?: \d+)?$", "", base_name)
    i = 0
    name = f"{base}{suffix}"
    field_lookup = {name_field: name}

    while queryset.filter(**field_lookup).exists():
        i += 1
        name = f"{base}{suffix} {i}"
        field_lookup[name_field] = name

    return name
