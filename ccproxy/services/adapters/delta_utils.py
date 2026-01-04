"""Delta accumulation utilities following OpenAI SDK patterns."""

from __future__ import annotations

from typing import Any


def accumulate_delta(
    accumulated: dict[str, Any], delta: dict[str, Any]
) -> dict[str, Any]:
    """Recursively merge delta into accumulated following OpenAI's rules.

    This function implements the same accumulation logic as OpenAI's SDK:
    - Concatenate strings
    - Add numbers (int/float)
    - Recursively merge dictionaries
    - Extend primitive lists
    - Merge object lists by 'index' key
    - Preserve 'index' and 'type' keys without modification

    Args:
        accumulated: The accumulated state to merge into
        delta: The delta to merge

    Returns:
        The merged result (may modify accumulated in-place)

    Raises:
        TypeError: For unsupported data types
        ValueError: For invalid list structures
    """
    # Handle None/empty cases
    if not delta:
        return accumulated
    if not accumulated:
        return dict(delta)

    # Work on a copy to avoid mutating input
    result = dict(accumulated)

    for key, delta_value in delta.items():
        if key not in result:
            # New key, just set it
            result[key] = delta_value
            continue

        current_value = result[key]

        # Handle different data type combinations
        if isinstance(current_value, str) and isinstance(delta_value, str):
            # Concatenate strings
            result[key] = current_value + delta_value

        elif isinstance(current_value, int | float) and isinstance(
            delta_value, int | float
        ):
            # Add numbers
            result[key] = current_value + delta_value

        elif isinstance(current_value, dict) and isinstance(delta_value, dict):
            # Recursively merge dictionaries
            result[key] = accumulate_delta(current_value, delta_value)

        elif isinstance(current_value, list) and isinstance(delta_value, list):
            # Handle list merging
            result[key] = _accumulate_list(current_value, delta_value)

        else:
            # For any other case, delta value overwrites
            result[key] = delta_value

    return result


def _accumulate_list(current: list[Any], delta: list[Any]) -> list[Any]:
    """Accumulate list values following OpenAI's patterns.

    - For primitive lists: extend
    - For object lists: merge by 'index' key

    Args:
        current: Current list value
        delta: Delta list value

    Returns:
        Merged list

    Raises:
        ValueError: If object list entries are missing required 'index' key
    """
    if not delta:
        return current
    if not current:
        return list(delta)

    # Check if this is an object list (contains dicts with 'index')
    has_indexed_objects = any(
        isinstance(item, dict) and "index" in item for item in (current + delta)
    )

    if not has_indexed_objects:
        # Primitive list - just extend
        return current + delta

    # Object list - merge by index
    result = list(current)

    for delta_item in delta:
        if not isinstance(delta_item, dict):
            # Mixed list types - append non-dict items
            result.append(delta_item)
            continue

        if "index" not in delta_item:
            raise ValueError("Dictionary in list delta must have 'index' key")

        delta_index = delta_item["index"]

        # Find existing item with same index
        existing_item = None
        existing_pos = None
        for i, item in enumerate(result):
            if isinstance(item, dict) and item.get("index") == delta_index:
                existing_item = item
                existing_pos = i
                break

        if existing_item is not None and existing_pos is not None:
            # Merge with existing item, preserving special keys
            merged = accumulate_delta(existing_item, delta_item)

            # Preserve 'index' and 'type' from original if not in delta
            for special_key in ["index", "type"]:
                if special_key not in delta_item and special_key in existing_item:
                    merged[special_key] = existing_item[special_key]

            result[existing_pos] = merged
        else:
            # New item - append to list
            result.append(delta_item)

    return result
