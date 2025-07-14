# cloudglide/event.py
import itertools
from collections import namedtuple

_counter = itertools.count()
Event = namedtuple("Event", ["time", "counter", "job", "etype"])


def next_event_counter():
    return next(_counter)
