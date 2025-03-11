def add_tabs(text: str) -> str:
    """Add a tab character to the start of each line in the text."""
    return "\n\t".join(text.split("\n"))
