"""Logger."""


class Log:  # pylint: disable=too-few-public-methods
    """Logger."""
    VERBOSE = False

    @staticmethod
    def log(*args):
        """Log arguments."""
        if Log.VERBOSE:
            print(*args)
