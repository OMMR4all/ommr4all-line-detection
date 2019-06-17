from abc import ABC, abstractmethod
from datetime import datetime
from multiprocessing import Process, Value, Lock


class LineDetectionCallback(ABC):
    def __init__(self, steps_per_page=8, total_pages=1):
        super().__init__()
        self.__total_pages = total_pages
        self.__state = Value('i', 0)
        self.__steps_per_page = steps_per_page
        self.__total_steps = steps_per_page * total_pages
        self.__processed_pages = Value('i', 0)
        self.__lock = Lock()

    def get_progress(self):
        return self.__state.value / self.__total_steps

    def get_total_pages(self):
        return self.__total_pages

    def get_processed_pages(self):
        return self.__processed_pages.value

    def update_total_state(self):
        with self.__lock:
            self.__state.value += 1
        self.changed()

    def update_page_counter(self):
        with self.__lock:
            self.__processed_pages.value += 1
        self.changed()

    def set_total_pages(self, value):
        self.__total_pages = value
        self.__total_steps = self.__steps_per_page * self.__total_pages

    @abstractmethod
    def changed(self):
        pass


class DummyLineDetectionCallback(LineDetectionCallback):
    def changed(self):
        print("Total progress: {} Page progress: {}".format(self.get_progress(), self.get_processed_pages()))
        print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])

        pass

    def __init__(self, total_steps=7, total_pages=1):
        super().__init__(total_steps, total_pages)

