import threading
from typing import Callable, Optional

observable_list = []


class Observable:
    def __init__(self, value=None):
        self.observers = {}
        self.value = value
        self.flag = threading.Event()
        observable_list.append(self)

    def subscribe(self, observer, callback: Callable[[Optional[any]], None] = None):
        if observer in self.observers:
            del self.observers[observer]
        self.observers[observer] = callback

    def notify(self):
        for observer in self.observers:
            self.observers[observer](self.value)

    def set_value(self, value):
        self.value = value
        self.flag.set()

    def get_value(self):
        return self.value


def notify_thread():
    while True:
        for observable in observable_list:
            if observable.flag.is_set():
                observable.notify()
                observable.flag.clear()
