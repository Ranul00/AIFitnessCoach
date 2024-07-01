
import customtkinter
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from test_bicep_curl_copy import (bicep_detection)
from squat import (squat_detection)
from front_raise import (front_raise_detection)

# ip camera
# cap = cv2.VideoCapture('http://192.168.1.6:8080/video')
cap = cv2.VideoCapture(1)

def onClickBack():
    cap.release()
    exercise_label.configure(text="")
    label.configure(image=None)



# Define the function to capture and display video
def capture_video(exercise = ""):

    # Capture a frame from the video feed
    ret, frame = cap.read()

    if exercise == "bicep_curl":
        exercise_label.configure(text="Bicep Curl")
        opencv_image = bicep_detection(cap=cap, image=frame, cv2=cv2)
    elif exercise == "squat":
        exercise_label.configure(text="Squat")
        opencv_image = squat_detection(cap=cap, image=frame, cv2=cv2)
    elif exercise == "front_raise":
        exercise_label.configure(text="Front Raise")
        opencv_image = front_raise_detection(cap=cap, image=frame, cv2=cv2)

    # Convert the frame from BGR to RGB
    # opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # opencv_image = squat_detection(cap=cap, image=frame, cv2=cv2)
    # opencv_image = front_raise_detection(cap=cap, image=frame, cv2=cv2)

    # Convert the frame to a PIL ImageTk object
    image = Image.fromarray(opencv_image)
    photo = ImageTk.PhotoImage(image)

    # Update the label with the new frame
    label.configure(image=photo)
    label.image = photo
    label.after(5,  lambda: capture_video(exercise))


# ======================GUI======================
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")

root = customtkinter.CTk()
root.geometry("1366x1080")
root.title("AI Fitness Trainer")




bicep_label = customtkinter.CTkLabel(
    master=root, text="AI Fitness Trainer", font=("Roboto", 36))
bicep_label.pack(pady=12, padx=10)

frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=70, fill="both", expand=True)

bicep_label = customtkinter.CTkLabel(
    master=frame, text="Pick The Exercise", font=("Roboto", 24))
bicep_label.pack(pady=12, padx=10)

bicep_button = customtkinter.CTkButton(
    master=frame, text="Bicep Curl", font=("Roboto", 18), command=lambda: capture_video(exercise="bicep_curl"))
bicep_button.pack(pady=12, padx=10)

squat_button = customtkinter.CTkButton(
    master=frame, text="Squat", font=("Roboto", 18), command=lambda: capture_video(exercise="squat"))
squat_button.pack(pady=12, padx=10)

front_raise_button = customtkinter.CTkButton(
    master=frame, text="Front Raise", font=("Roboto", 18), command=lambda: capture_video(exercise="front_raise"))
front_raise_button.pack(pady=12, padx=10)

video_frame = customtkinter.CTkScrollableFrame(master=root,  width=800, height=600)
video_frame.pack(pady=20, padx=70, fill="both", expand=True)

exercise_label = customtkinter.CTkLabel(
    master=video_frame, text="", font=("Roboto", 36))
exercise_label.pack(pady=12, padx=10)

label = tk.Label(master=video_frame)
label.pack()

back_button = customtkinter.CTkButton(
master=video_frame, text="Back", font=("Roboto", 18), command=onClickBack)
back_button.pack(pady=12, padx=2)


root.mainloop()


