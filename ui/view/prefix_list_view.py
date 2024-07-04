import tkinter as tk
from typing import Callable, Optional
from ui.state import RootState
from ui.util.listutil import *


class PrefixListView:

    def __init__(self, window: tk.Widget, state: RootState):
        self.root = tk.Frame(window)
        self.root.pack(fill="x", expand=True)
        self.state = state

        self.__define_ui()
        self.__define_event()
        self.__define_state_update()

    def __define_ui(self):

        self.list_image_files = tk.Listbox(self.root)
        self.list_image_files_process = tk.Listbox(self.root)
        self.list_image_files.pack(side=tk.LEFT, fill="x", expand=True)
        self.list_image_files_process.pack(side=tk.LEFT, fill="x", expand=True)
        pass

    def __define_event(self):

        set_list_on_select(
            self.list_image_files,
            lambda value: (
                append_list_distinct(
                    self.state.preprocessing.parameter.IMAGE_FILES, value
                ),
                self.state.image_label_list.set_value(
                    self.state.preprocessing.parameter.IMAGE_FILES
                ),
            ),
        )
        set_list_on_select(
            self.list_image_files_process,
            lambda value: (
                remove_list(self.state.preprocessing.parameter.IMAGE_FILES, value),
                self.state.image_label_list.set_value(
                    self.state.preprocessing.parameter.IMAGE_FILES
                ),
            ),
        ),

    def __define_state_update(self):
        self.state.data_label_dict.subscribe(
            self.list_image_files,
            lambda value: (
                self.list_image_files.delete(0, tk.END),
                self.list_image_files.insert(0, *value.keys()),
            ),
        )

        self.state.image_label_list.subscribe(
            self.list_image_files_process,
            lambda value: (
                self.list_image_files_process.delete(0, tk.END),
                self.list_image_files_process.insert(0, *value),
            ),
        )

    def __call__(self):
        return self.root
