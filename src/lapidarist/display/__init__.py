import logging

from rich.text import Text

logging.getLogger(__name__).addHandler(logging.NullHandler())


def header() -> Text:
    text = Text("""[bold]Lapidarist[/bold] 💎, The AI Alliance""")
    # TODO version, timestamp, ...
    return text
