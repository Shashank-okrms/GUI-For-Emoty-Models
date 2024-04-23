import os
import numpy as np
import librosa
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image, ImageTk
    
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS2
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Load speech prediction model
loaded_model = load_model(resource_path('Speech_model.h5'))

# Load emotion detection model
face_classifier = cv2.CascadeClassifier(resource_path('haarcascade_frontalface_default.xml'))
emotion_model = load_model(resource_path('Emotion_model.h5'))
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load drowsiness detection model
drowsiness_model = load_model(resource_path('Drowsiness_model.h5'))

class EmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emoty")
        self.selected_image = None
        self.detected_result = ""

        self.root.configure(bg='#263D42')

        # Create frame for the icon and title
        self.icon_frame = tk.Frame(root, bg='#263D42')
        self.icon_frame.pack(side=tk.TOP, pady=10)

        # Create frame for the buttons
        self.button_frame = tk.Frame(root, bg='#263D42')
        self.button_frame.pack(side=tk.TOP, pady=10)

        # Create frame for the result
        self.result_frame = tk.Frame(root, bg='#263D42')
        self.result_frame.pack(side=tk.TOP, pady=10)

        # Draw icon and title
        self.draw_icon_and_title()

        # Create buttons
        self.create_buttons()

    def draw_icon_and_title(self):
        icon_image = Image.open('icon.ico')
        icon_image = icon_image.resize((100, 100), resample=Image.LANCZOS)
        self.icon_photo = ImageTk.PhotoImage(icon_image)

        self.icon_frame.config(highlightthickness=0)

        icon_label = tk.Label(self.icon_frame, image=self.icon_photo, bg='#263D42')
        icon_label.pack(side=tk.TOP)

        title_text = "Emoty"
        title_label = tk.Label(self.icon_frame, text=title_text, font=("Segoe Script", 25), fg='white', bg='#263D42')
        title_label.pack(side=tk.TOP)

    def create_buttons(self):
        buttons_info = [
            ("Via Image_Emotion", self.predict_image_emotion),
            ("Via Video_Emotion", self.predict_emotion_video),
            ("Via Speech", self.predict_speech_emotion),
            ("Via Image_Drowsiness", self.predict_image_drowsiness),
            ("Refresh", self.refresh)
        ]

        for text, command in buttons_info:
            button = tk.Button(self.button_frame, text=text, command=command, fg='white',bg="#263D42",font=('Garamond',15,'bold'))
            button.pack(side=tk.LEFT, padx=5, pady=10)

    def predict_speech_emotion(self):
        self.refresh()
        self.selected_image = None  # Set selected_image to None when performing speech recognition
        audio_file_path = filedialog.askopenfilename(title="Select Audio File", filetypes=[("Audio files", "*.wav")])
        if audio_file_path:
            predicted_emotion = self.predict_emotion(audio_file_path)
            self.detected_result = predicted_emotion
            self.display_result()


    def predict_image_emotion(self):
        self.refresh()
        image_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if image_path:
            input_image = cv2.imread(image_path)
            if input_image is None:
                messagebox.showerror("Error", "Unable to read the image.")
                return
            self.selected_image = Image.open(image_path)
            detected_emotions = self.detect_emotion(input_image)
            self.detected_result = self.calculate_average_emotion(detected_emotions)
            self.display_result()

    def predict_image_drowsiness(self):
        self.refresh()
        image_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if image_path:
            self.selected_image = Image.open(image_path)
            prediction = self.predict_drowsiness(image_path)
            self.detected_result = prediction
            self.display_result()

    def predict_emotion_video(self):
        self.refresh()
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Unable to capture frame")
                break

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces = face_classifier.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                # Extract the face region from the frame
                roi_gray = gray[y:y+h, x:x+w]

                # Resize the face region to match the input size of the model
                roi_gray = cv2.resize(roi_gray, (224, 224), interpolation=cv2.INTER_AREA)
                roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)

                # Normalize the face region
                roi = roi_color.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Predict the emotion using the loaded model
                prediction = emotion_model.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]

                # Display the predicted emotion on the frame
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

            # Display the frame with the detected emotions
            cv2.imshow('Emotion Detector', frame)

            # Exit the loop if 'q' is pressed or camera window is closed
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Emotion Detector', cv2.WND_PROP_VISIBLE) < 1:
                break

        # Release the webcam and close all windows
        cap.release()
        cv2.destroyAllWindows()

    def predict_emotion(self, audio_file):
        mfcc_features = self.extract_mfcc_from_audio(audio_file)
        preprocessed_features = self.preprocess_features(mfcc_features)
        predicted_emotion = loaded_model.predict(preprocessed_features)
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Pleasant Surprise', 'Sad']
        predicted_emotion_label = emotions[np.argmax(predicted_emotion)]
        return predicted_emotion_label

    def detect_emotion(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)
        detected_emotions = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (224, 224), interpolation=cv2.INTER_AREA)
            roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
            roi = roi_color.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            prediction = emotion_model.predict(roi)[0]
            label = emotion_labels[np.argmax(prediction)]
            detected_emotions.append(label)
        return detected_emotions

    def calculate_average_emotion(self, emotions):
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        most_frequent_emotion = max(emotion_counts, key=emotion_counts.get)
        return most_frequent_emotion

    def preprocess_features(self, features):
        features = np.expand_dims(features, axis=0)
        features = np.expand_dims(features, axis=-1)
        return features

    def extract_mfcc_from_audio(self, audio_file):
        y, sr = librosa.load(audio_file, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfcc

    def preprocess_image(self, image_path):
        image_size = (80, 80)
        img = Image.open(image_path).resize(image_size)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img = np.array(img) / 255.0
        return img

    def predict_drowsiness(self, image_path):
        img = self.preprocess_image(image_path)
        result = drowsiness_model.predict(img[np.newaxis, ...])
        predicted_label_index = np.argmax(result)
        if predicted_label_index in [0, 3]:
            return 'Drowsiness Detected'
        elif predicted_label_index == 1:
            return 'No Drowsiness Detected'
        elif predicted_label_index == 2:
            return 'No Drowsiness Detected'

    def display_result(self):
        self.result_frame.destroy()
        self.result_frame = tk.Frame(root, bg='#263D42')
        self.result_frame.pack(side=tk.TOP, pady=10)

        if self.detected_result != "Via Speech" and self.selected_image:  # Check if it's not speech recognition and an image is selected
            self.selected_image.thumbnail((300, 300))
            self.image_label = ImageTk.PhotoImage(self.selected_image)
            image_label = tk.Label(self.result_frame, image=self.image_label, bg='#263D42')
            image_label.image = self.image_label  # Keep a reference to prevent garbage collection
            image_label.pack(side=tk.TOP, padx=10, pady=10)
        if self.detected_result:
            result_label = tk.Label(self.result_frame, text=self.detected_result, font=("Helvetica", 12), fg="white", bg='#263D42')
            result_label.pack(side=tk.TOP, padx=10, pady=10)



    def refresh(self):
        self.result_frame.destroy()
        self.result_frame = tk.Frame(self.root, bg='#263D42')
        self.result_frame.pack(side=tk.TOP, pady=10)

if __name__ == "__main__":
    root = tk.Tk()

    # Set window icon
    icon_path = resource_path('icon.ico')
    if os.path.exists(icon_path):
        root.iconbitmap(icon_path)

    app = EmotionDetectionApp(root)
    root.mainloop()