from typing import Callable
import tkinter as tk


def set_list_on_select(listbox: tk.Listbox, callback: Callable):
    def on_select(evt):
        try:
            w = evt.widget
            index = int(w.curselection()[0])
            value = w.get(index)
            callback(value)
        except IndexError:
            pass

    listbox.bind("<<ListboxSelect>>", on_select)


def append_list_distinct(l: list, value):
    if value not in l:
        l.append(value)


def remove_list(l: list, value):
    if value in l:
        l.remove(value)
