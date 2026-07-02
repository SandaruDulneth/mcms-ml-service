import logging


def configure_logging() -> None:
    """Set a consistent console log format for this application and its libraries."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
