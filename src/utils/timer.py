import time


class Timer:
    def __init__(self, name):
        self.name = name
        self.start = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start
        if elapsed < 1:
            print("%s: %d ms" % (self.name, elapsed * 1000))
        elif elapsed < 60:
            print("%s: %d s %d ms" % (self.name, elapsed, elapsed * 1000 % 1000))
        else:
            print("%s: %d m %d s %d ms" % (self.name, elapsed // 60, elapsed % 60, elapsed * 1000 % 1000))
        return False
