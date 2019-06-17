from abc import ABC, abstractmethod
from datetime import datetime


class LineDetectionCallback(ABC):
    def __init__(self, steps_per_page=8, total_pages=1):
        super().__init__()
        self.__total_pages = total_pages
        self.__state = 0
        self.__steps_per_page = steps_per_page
        self.__total_steps = steps_per_page * total_pages
        self.__processed_pages = 0

    def get_progress(self):
        return self.__state / self.__total_steps

    def get_total_pages(self):
        return self.__total_pages

    def get_processed_pages(self):
        return self.__processed_pages

    def update_total_state(self):
        self.__state += 1
        self.changed()

    def update_page_counter(self):
        self.__processed_pages += 1
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

