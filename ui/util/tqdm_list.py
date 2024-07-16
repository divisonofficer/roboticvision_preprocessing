from typing import Optional, Callable


class TQDMList:
    def __init__(self, data: list, prefix: Optional[str], callback: Callable):
        self.data = data
        self.callback = callback
        if prefix is None:
            self.prefix = ""
        else:
            self.prefix = prefix

    def __iter__(self):
        for idx, item in enumerate(self.data):
            yield item
            self.callback(f"{self.prefix}{idx}/{len(self.data)}", idx / len(self.data))
        self.callback(f"{self.prefix}{len(self.data)}/{len(self.data)}", 1)
