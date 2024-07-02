from ui.observable import Observable


class RootState:
    def __init__(self):
        self.data_folder = Observable()

        self.data_capture_count = Observable(0)
        self.data_scene_count = Observable(0)
        self.data_label_dict = Observable({})

        self.data_examine = Observable({})

        self.image_label_list = Observable([])

        self.tqdm_output = Observable("")
        self.tqdm_progress = Observable(0)
