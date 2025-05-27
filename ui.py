# ui.py

import tkinter as tk
from tkinter import messagebox
from attendance_system import start_recognition

def run_app():
    root = tk.Tk()
    root.title("Face Attendance System")
    root.geometry("500x400")
    root.config(bg="#2E3B4E")  # Dark background

    title_label = tk.Label(root, text="Face Attendance System", font=("Helvetica", 20, "bold"), fg="#FFD700", bg="#2E3B4E")
    title_label.pack(pady=30)

    feedback_label = tk.Label(root, text="Ready to start attendance!", font=("Arial", 14), fg="#FFFFFF", bg="#2E3B4E")
    feedback_label.pack(pady=10)

    def update_feedback(text):
        feedback_label.config(text=text)

    def on_enter(e):
        btn.config(bg="#5D9CEC")

    def on_leave(e):
        btn.config(bg="#3A87AD")

    def start_attendance():
        update_feedback("Attendance in progress...")
        start_recognition()
        update_feedback("Attendance complete!")
        messagebox.showinfo("Success", "Attendance marked successfully!")

    btn = tk.Button(root, text="Start Attendance", font=("Arial", 14), command=start_attendance, 
                    fg="white", bg="#3A87AD", relief="flat", height=2, width=20)
    btn.pack(pady=30)
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

    footer_label = tk.Label(root, text="Face Recognition Powered by Python & dlib", font=("Arial", 10), fg="#A9A9A9", bg="#2E3B4E")
    footer_label.pack(side=tk.BOTTOM, pady=15)

    root.mainloop()

if __name__ == "__main__":
    run_app()
