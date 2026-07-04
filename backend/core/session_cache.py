from functools import lru_cache

import fastf1


@lru_cache(maxsize=16)
def get_session(year: int, round_number: int, session_type: str):
    """Load (and cache) a fastf1 session, since loading is expensive and
    the resulting object isn't JSON-serializable to pass around otherwise."""
    session = fastf1.get_session(year, round_number, session_type)
    session.load()
    return session
