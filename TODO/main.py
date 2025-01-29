import tkinter as tk
from tkinter import messagebox

# Function to add a task
def add_task():
    task = entry_task.get().strip()
    if task:
        listbox_tasks.insert(tk.END, task)
        entry_task.delete(0, tk.END)
    else:
        messagebox.showwarning("Warning", "Task cannot be empty!")

# Function to mark a task as completed
def mark_completed():
    try:
        selected_task_index = listbox_tasks.curselection()[0]
        task = listbox_tasks.get(selected_task_index)
        listbox_tasks.delete(selected_task_index)
        listbox_tasks.insert(selected_task_index, f"✓ {task}")
    except IndexError:
        messagebox.showwarning("Warning", "Please select a task to mark as completed!")

# Function to delete a task
def delete_task():
    try:
        selected_task_index = listbox_tasks.curselection()[0]
        listbox_tasks.delete(selected_task_index)
    except IndexError:
        messagebox.showwarning("Warning", "Please select a task to delete!")

# Function to clear all tasks
def clear_tasks():
    listbox_tasks.delete(0, tk.END)

# Main application window
app = tk.Tk()
app.title("To-Do List App")
app.geometry("400x400")
app.resizable(False, False)

# Task Entry
entry_task = tk.Entry(app, width=40, font=("Arial", 12))
entry_task.pack(pady=10)

# Buttons
button_add = tk.Button(app, text="Add Task", width=20, command=add_task)
button_add.pack(pady=5)

button_mark = tk.Button(app, text="Mark as Completed", width=20, command=mark_completed)
button_mark.pack(pady=5)

button_delete = tk.Button(app, text="Delete Task", width=20, command=delete_task)
button_delete.pack(pady=5)

button_clear = tk.Button(app, text="Clear All Tasks", width=20, command=clear_tasks)
button_clear.pack(pady=5)

# Task Listbox
listbox_tasks = tk.Listbox(app, width=50, height=10, font=("Arial", 12))
listbox_tasks.pack(pady=10)

# Run the application
app.mainloop()