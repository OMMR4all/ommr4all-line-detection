

class LineDetectionCallback:
    def __init__(self):
        super().__init__()
        self.state = 0
        self.total_pages = 1
        self.page_state = 0
        self.total_steps = 12

    def init(self, state, page_state, total_steps, total_pages):
        self.total_pages = total_pages
        self.state = state
        self.page_state = page_state
        self.total_steps = total_steps

    def get_progress(self):
        return self.state / self.total_pages

    def get_current_page_progress(self):
        return self.page_state / self.total_steps

    def update_current_page_state(self, state = None):
        if state is not None:
            self.page_state = 0
        else:
            self.page_state += 1


