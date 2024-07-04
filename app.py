import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from preprocessing import Preprocessing
from typing import Optional, Callable
from PIL import Image, ImageDraw, ImageFont, ImageTk
from ui.observable import Observable, notify_thread
import threading
from ui.state import RootState
from ui.view.prefix_list_view import PrefixListView
from colmap_wrap import ColmapWrap

preprocessing: Preprocessing
colmap: ColmapWrap


state: RootState


def emoji(emoji, size=32):
    # convert emoji to CTkImage
    font = ImageFont.truetype("NotoColorEmoji.ttf", size=size)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((size / 2, size / 2), emoji, embedded_color=True, font=font, anchor="mm")
    return ImageTk.PhotoImage(img)


def open_file_dialog(store: Optional[Observable]):
    folder = filedialog.askdirectory()
    if folder:
        if store:
            store.set_value(folder)


def ui_preprocessing_args(window: tk.Tk):
    ###############################
    # Callback
    ##################################
    def load_gzip_dialog():
        file = filedialog.askopenfilename(
            filetypes=[("Gzip files", "*.gz")],
            title="Select a Gzip file",
        )
        if file:
            data = preprocessing.load_gzip(file)
            state.data_folder.set_value(data)

    ###############################
    # Widgets
    ##################################
    frame_btn_folder = tk.Frame(window)

    btn_load_folder = tk.Button(
        frame_btn_folder,
        text="Open",
        command=lambda: open_file_dialog(
            state.data_folder,
        ),
    )

    btn_load_folder.grid(row=0, column=0)

    btn_load_gzip = tk.Button(
        frame_btn_folder,
        text="Load Gzip",
        command=load_gzip_dialog,
    )

    label_folder = tk.Label(frame_btn_folder, text="path")

    btn_load_gzip.grid(row=0, column=1)
    label_folder.grid(row=0, column=2)

    frame_btn_folder.pack(fill="x")

    ###############################
    # State Subscription
    ##################################

    state.data_folder.subscribe(
        btn_load_folder,
        lambda value: (
            label_folder.config(text=value),
            preprocessing.parameter.__setattr__("FOLDER", value),
            state.data_examine.set_value(preprocessing.examine_folder()),
            state.image_label_list.set_value(preprocessing.parameter.IMAGE_FILES),
            colmap.set_space_folder(value),
        ),
    )

    state.data_examine.subscribe(
        btn_load_folder,
        lambda value: (
            state.data_capture_count.set_value(value["capture_cnt"]),
            state.data_scene_count.set_value(value["scene_cnt"]),
            state.data_label_dict.set_value(value["key_found"]),
        ),
    )


def ui_tqdm_progress(window: tk.Tk):
    frame = tk.Frame(window)
    progress_label = tk.Label(frame, text="Progress: ")
    progress_label.pack(side=tk.LEFT)
    progress_bar = ttk.Progressbar(
        frame, orient="horizontal", length=100, mode="determinate"
    )
    progress_bar.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    state.tqdm_progress.subscribe(
        progress_bar,
        lambda value: progress_bar.config(value=value * 100),
    )
    state.tqdm_output.subscribe(
        progress_label,
        lambda value: progress_label.config(text=value),
    )
    frame.pack(fill="x")


def ui_data_examine(window: tk.Tk):

    PrefixListView(window, state)

    label_capture_count = tk.Label(window, text="Capture Count: ")
    label_scene_count = tk.Label(window, text="Scene Count: ")
    label_capture_count.pack()
    label_scene_count.pack()

    state.data_capture_count.subscribe(
        label_capture_count,
        lambda value: label_capture_count.config(text=f"Capture Count: {value}"),
    )

    state.data_scene_count.subscribe(
        label_scene_count,
        lambda value: label_scene_count.config(text=f"Scene Count: {value}"),
    )

    btn_frame = tk.Frame(window)
    btn_frame.pack(fill="x")
    btn_combine = tk.Button(
        btn_frame,
        text="Combine",
        command=lambda: preprocessing.group_images(),
    )

    btn_combine.pack(side=tk.LEFT)

    def hdr_fusion():
        def hdr_fusion_thread():
            preprocessing.hdr_fusion_space()
            state.data_examine.set_value(preprocessing.examine_folder())

        thread = threading.Thread(target=hdr_fusion_thread)
        thread.start()

    btn_hdr_fusion = tk.Button(
        btn_frame,
        text="HDR Fusion",
        command=hdr_fusion,
    )
    btn_hdr_fusion.pack(side=tk.LEFT)


def ui_colmap(window: tk.Tk):
    frame = tk.Frame(window)
    frame.pack(fill="x")

    btn_run_all = tk.Button(
        frame,
        text="Colmap All",
        command=lambda: colmap.run_fast_mapper(),
    )
    btn_run_all.pack(side=tk.LEFT)
    btn_run_colmap = tk.Button(
        frame,
        text="Colmap FE",
        command=lambda: colmap.run_feature_extractor(),
    )
    btn_run_colmap.pack(side=tk.LEFT)

    btn_run_fm = tk.Button(
        frame,
        text="Colmap FM",
        command=lambda: colmap.run_feature_matcher(),
    )
    btn_run_fm.pack(side=tk.LEFT)

    btn_run_mapper = tk.Button(
        frame,
        text="Colmap Mapper",
        command=lambda: colmap.run_feature_mapper(),
    )
    btn_run_mapper.pack(side=tk.LEFT)
    btn_run_converter = tk.Button(
        frame,
        text="Colmap Converter",
        command=lambda: colmap.run_model_convert_thread(),
    )

    btn_run_converter.pack(side=tk.LEFT)


def main():
    window = tk.Tk()
    window.title("Preprocessing")
    window.geometry("720x480")
    window.resizable(True, True)

    ui_preprocessing_args(window)
    ui_data_examine(window)
    ui_colmap(window)
    ui_tqdm_progress(window)
    window.mainloop()


if __name__ == "__main__":
    preprocessing = Preprocessing(
        None,
        lambda x, y: (state.tqdm_output.set_value(x), state.tqdm_progress.set_value(y)),
    )
    state = RootState(preprocessing)
    colmap = ColmapWrap(
        lambda x, y: (state.tqdm_output.set_value(x), state.tqdm_progress.set_value(y)),
    )
    thread = threading.Thread(target=notify_thread)
    thread.start()
    main()
    thread.join()
