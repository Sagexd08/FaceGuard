import cv2
from cv2 import face
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle

class FaceAttendanceSystem:
    def __init__(self):
        self.known_faces_dir = "known_faces"
        self.training_data_dir = "training_data"
        self.attendance_file = "attendance.csv"
        self.recognizer = face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.user_dict = {}
        
        for directory in [self.known_faces_dir, self.training_data_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        self.load_trained_data()

    def load_trained_data(self):
        model_file = "face_recognizer_model.yml"
        user_file = "user_dict.pkl"
        
        if os.path.exists(model_file) and os.path.exists(user_file):
            self.recognizer.read(model_file)
            with open(user_file, 'rb') as f:
                self.user_dict = pickle.load(f)
            print("Loaded existing face recognition model")

    def save_trained_data(self):
        self.recognizer.write("face_recognizer_model.yml")
        with open("user_dict.pkl", 'wb') as f:
            pickle.dump(self.user_dict, f)

    def register_new_face(self):
        name = input("Enter name for registration: ")
        if name.lower() == 'q':
            return
        
        user_id = len(self.user_dict)
        self.user_dict[user_id] = name
        
        print("Starting face capture. Press 'q' to quit.")
        print("Please move your face slightly to capture different angles.")
        
        cap = cv2.VideoCapture(0)
        face_count = 0
        max_faces = 50
        
        while face_count < max_faces:
            ret, frame = cap.read()
            if not ret:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face_count += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_img = gray[y:y+h, x:x+w]
                img_path = os.path.join(self.training_data_dir, f"{name}_{face_count}.jpg")
                cv2.imwrite(img_path, face_img)
            
            cv2.putText(frame, f"Captured: {face_count}/{max_faces}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Face Registration", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q') or face_count >= max_faces:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        self.train_model()
        print(f"Successfully registered {name}")

    def train_model(self):
        faces = []
        ids = []
        
        for filename in os.listdir(self.training_data_dir):
            if filename.endswith(".jpg"):
                img_path = os.path.join(self.training_data_dir, filename)
                name = filename.split('_')[0]
                
                for user_id, user_name in self.user_dict.items():
                    if user_name == name:
                        face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        faces.append(face_img)
                        ids.append(user_id)
                        break
        
        if faces and ids:
            self.recognizer.train(faces, np.array(ids))
            self.save_trained_data()

    def mark_attendance(self, name):
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")
        
        if not os.path.exists(self.attendance_file):
            df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
            df.to_csv(self.attendance_file, index=False)
        
        df = pd.read_csv(self.attendance_file)
        
        if not df[(df['Name'] == name) & (df['Date'] == date)].empty:
            return False, "Attendance already marked for today"
        
        new_record = pd.DataFrame({'Name': [name], 'Date': [date], 'Time': [time]})
        df = pd.concat([df, new_record], ignore_index=True)
        df.to_csv(self.attendance_file, index=False)
        
        return True, f"Attendance marked for {name}"

    def start_attendance(self):
        if not self.user_dict:
            print("No registered users found. Please register faces first.")
            return
        
        print("Starting attendance system. Press 'q' to quit.")
        cap = cv2.VideoCapture(0)
        attendance_marked = set()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_img = gray[y:y+h, x:x+w]
                
                try:
                    id_, confidence = self.recognizer.predict(face_img)
                    if confidence < 100:
                        name = self.user_dict.get(id_, "Unknown")
                        confidence_text = f"{round(100 - confidence)}%"
                        
                        if name not in attendance_marked:
                            success, message = self.mark_attendance(name)
                            if success:
                                attendance_marked.add(name)
                                print(message)
                    else:
                        name = "Unknown"
                        confidence_text = ""
                    
                    cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    if confidence_text:
                        cv2.putText(frame, confidence_text, (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except:
                    pass
            
            cv2.imshow("Attendance System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def generate_report(self):
        if not os.path.exists(self.attendance_file):
            print("No attendance records found.")
            return
        
        df = pd.read_csv(self.attendance_file)
        
        print("\nAttendance Report:")
        print("=================")
        print(df.to_string(index=False))

    def display_menu(self):
        while True:
            print("\nFace Recognition Attendance System")
            print("=================================")
            print("1. Register New Face")
            print("2. Take Attendance")
            print("3. Generate Report")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ")
            
            if choice == '1':
                self.register_new_face()
            elif choice == '2':
                self.start_attendance()
            elif choice == '3':
                self.generate_report()
            elif choice == '4':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    system = FaceAttendanceSystem()
    system.display_menu()