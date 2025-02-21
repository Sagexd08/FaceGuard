import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import pickle
import shutil
import json
import logging
from typing import Tuple, Dict, List, Optional
import threading
import time

class FaceAttendanceSystem:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.known_faces_dir = os.path.join(self.base_dir, "known_faces")
        self.training_data_dir = os.path.join(self.base_dir, "training_data")
        self.backup_dir = os.path.join(self.base_dir, "backups")
        self.attendance_file = os.path.join(self.base_dir, "attendance.csv")
        self.log_file = os.path.join(self.base_dir, "system.log")
        self.user_meta_file = os.path.join(self.base_dir, "user_metadata.json")
        
        logging.basicConfig(filename=self.log_file, level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.user_dict = {}
        self.user_metadata = {}
        self.backup_thread = None
        self.last_backup_time = None
        
        self.confidence_threshold = 65
        self.min_face_size = (60, 60)
        self.max_faces = 50
        self.backup_interval = timedelta(hours=24)
        
        self._initialize_system()

    def _initialize_system(self):
        for directory in [self.known_faces_dir, self.training_data_dir, self.backup_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        self.load_trained_data()
        self.load_user_metadata()
        self.start_backup_thread()
        logging.info("System initialized successfully")

    def load_trained_data(self):
        model_file = os.path.join(self.base_dir, "face_recognizer_model.yml")
        user_file = os.path.join(self.base_dir, "user_dict.pkl")
        
        try:
            if os.path.exists(model_file) and os.path.exists(user_file):
                self.recognizer.read(model_file)
                with open(user_file, 'rb') as f:
                    self.user_dict = pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading trained data: {str(e)}")
            self.user_dict = {}

    def load_user_metadata(self):
        try:
            if os.path.exists(self.user_meta_file):
                with open(self.user_meta_file, 'r') as f:
                    self.user_metadata = json.load(f)
        except Exception as e:
            logging.error(f"Error loading user metadata: {str(e)}")
            self.user_metadata = {}

    def save_trained_data(self):
        try:
            self.recognizer.write(os.path.join(self.base_dir, "face_recognizer_model.yml"))
            with open(os.path.join(self.base_dir, "user_dict.pkl"), 'wb') as f:
                pickle.dump(self.user_dict, f)
            self.save_user_metadata()
        except Exception as e:
            logging.error(f"Error saving trained data: {str(e)}")

    def save_user_metadata(self):
        try:
            with open(self.user_meta_file, 'w') as f:
                json.dump(self.user_metadata, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving user metadata: {str(e)}")

    def create_backup(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"backup_{timestamp}")
        
        try:
            os.makedirs(backup_path)
            shutil.copytree(self.training_data_dir, os.path.join(backup_path, "training_data"))
            
            for file in ["face_recognizer_model.yml", "user_dict.pkl", "user_metadata.json", "attendance.csv"]:
                if os.path.exists(os.path.join(self.base_dir, file)):
                    shutil.copy2(os.path.join(self.base_dir, file), os.path.join(backup_path, file))
            
            self.last_backup_time = datetime.now()
            self._clean_old_backups()
            
        except Exception as e:
            logging.error(f"Backup creation failed: {str(e)}")

    def _clean_old_backups(self):
        try:
            backups = sorted([d for d in os.listdir(self.backup_dir) if d.startswith("backup_")])
            if len(backups) > 5:
                for old_backup in backups[:-5]:
                    shutil.rmtree(os.path.join(self.backup_dir, old_backup))
        except Exception as e:
            logging.error(f"Error cleaning old backups: {str(e)}")

    def start_backup_thread(self):
        def backup_worker():
            while True:
                if not self.last_backup_time or datetime.now() - self.last_backup_time >= self.backup_interval:
                    self.create_backup()
                time.sleep(3600)
        
        self.backup_thread = threading.Thread(target=backup_worker, daemon=True)
        self.backup_thread.start()

    def check_face_quality(self, face_img) -> Tuple[bool, str]:
        try:
            if face_img.shape[0] < self.min_face_size[0] or face_img.shape[1] < self.min_face_size[1]:
                return False, "Face too small"
            
            brightness = np.mean(face_img)
            if brightness < 40: return False, "Image too dark"
            if brightness > 250: return False, "Image too bright"
            
            contrast = np.std(face_img)
            if contrast < 20: return False, "Low contrast"
            
            return True, "OK"
            
        except Exception as e:
            logging.error(f"Error in face quality check: {str(e)}")
            return False, "Error in quality check"

    def register_new_face(self):
        try:
            name = input("Enter name for registration: ").strip()
            if not name or name.lower() == 'q': return
            
            email = input("Enter email (optional): ").strip()
            department = input("Enter department (optional): ").strip()
            
            user_id = max(self.user_dict.keys()) + 1 if self.user_dict else 0
            
            self.user_metadata[name] = {
                "email": email,
                "department": department,
                "registration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            print("\nStarting face capture. Press 'q' to quit, 'c' to capture, 'r' to retry")
            
            cap = cv2.VideoCapture(0)
            face_count = 0
            
            while face_count < self.max_faces:
                ret, frame = cap.read()
                if not ret: continue
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=self.min_face_size)
                
                frame_with_guide = frame.copy()
                h, w = frame.shape[:2]
                guide_size = min(h, w) // 2
                guide_x = (w - guide_size) // 2
                guide_y = (h - guide_size) // 2
                cv2.rectangle(frame_with_guide, (guide_x, guide_y),
                            (guide_x + guide_size, guide_y + guide_size),
                            (255, 255, 255), 2)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame_with_guide, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    face_img = gray[y:y+h, x:x+w]
                    
                    quality_ok, message = self.check_face_quality(face_img)
                    color = (0, 255, 0) if quality_ok else (0, 0, 255)
                    cv2.putText(frame_with_guide, message, (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                cv2.putText(frame_with_guide, f"Captured: {face_count}/{self.max_faces}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Face Registration", frame_with_guide)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('c') and len(faces) > 0:
                    face_img = gray[y:y+h, x:x+w]
                    quality_ok, _ = self.check_face_quality(face_img)
                    
                    if quality_ok:
                        img_path = os.path.join(self.training_data_dir, f"{name}_{face_count}.jpg")
                        cv2.imwrite(img_path, face_img)
                        face_count += 1
            
            cap.release()
            cv2.destroyAllWindows()
            
            if face_count > 0:
                self.user_dict[user_id] = name
                self.train_model()
                self.save_trained_data()
                print(f"\nSuccessfully registered {name}")
                logging.info(f"New user registered: {name} (ID: {user_id})")
            else:
                print("\nRegistration cancelled - no images captured")
                
        except Exception as e:
            logging.error(f"Error in face registration: {str(e)}")
            print("An error occurred during registration")

    def delete_face(self):
        if not self.user_dict:
            print("No registered users found.")
            return
        
        try:
            print("\nRegistered Users:")
            print("================")
            for user_id, name in self.user_dict.items():
                metadata = self.user_metadata.get(name, {})
                department = metadata.get('department', 'N/A')
                reg_date = metadata.get('registration_date', 'N/A')
                print(f"ID: {user_id} | Name: {name} | Department: {department} | Registered: {reg_date}")
            
            user_id = input("\nEnter the ID of the user to delete (or -1 to cancel): ")
            if not user_id.isdigit() or int(user_id) == -1:
                return
            
            user_id = int(user_id)
            if user_id not in self.user_dict:
                print("Invalid user ID")
                return
            
            name = self.user_dict[user_id]
            confirmation = input(f"\nAre you sure you want to delete {name}? This cannot be undone. (yes/no): ")
            
            if confirmation.lower() != 'yes':
                print("Deletion cancelled")
                return
            
            self.create_backup()
            
            for filename in os.listdir(self.training_data_dir):
                if filename.startswith(f"{name}_"):
                    os.remove(os.path.join(self.training_data_dir, filename))
            
            del self.user_dict[user_id]
            if name in self.user_metadata:
                del self.user_metadata[name]
            
            if self.user_dict:
                self.train_model()
            else:
                model_file = os.path.join(self.base_dir, "face_recognizer_model.yml")
                if os.path.exists(model_file):
                    os.remove(model_file)
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            self.save_trained_data()
            
            if os.path.exists(self.attendance_file):
                df = pd.read_csv(self.attendance_file)
                df = df[df['Name'] != name]
                df.to_csv(self.attendance_file, index=False)
            
            print(f"\nSuccessfully deleted user: {name}")
            logging.info(f"User deleted: {name} (ID: {user_id})")
            
        except Exception as e:
            logging.error(f"Error in face deletion: {str(e)}")
            print("An error occurred during deletion")

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
            if not ret: continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=self.min_face_size)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_img = gray[y:y+h, x:x+w]
                
                try:
                    id_, confidence = self.recognizer.predict(face_img)
                    if confidence < self.confidence_threshold:
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
                except Exception as e:
                    logging.error(f"Error during face recognition: {str(e)}")
            
            cv2.imshow("Attendance System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def generate_report(self):
        try:
            if not os.path.exists(self.attendance_file):
                print("No attendance records found.")
                return
            
            df = pd.read_csv(self.attendance_file)
            if df.empty:
                print("No attendance records found.")
                return
            
            print("\nAttendance Report Options:")
            print("1. Daily Report")
            print("2. Weekly Report")
            print("3. Monthly Report")
            print("4. Custom Date Range")
            print("5. By Department")
            
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == '1':
                date = input("Enter date (YYYY-MM-DD) or press Enter for today: ")
                if not date:
                    date = datetime.now().strftime("%Y-%m-%d")
                daily_df = df[df['Date'] == date]
                if daily_df.empty:
                    print(f"No records found for {date}")
                else:
                    print(f"\nDaily Attendance Report for {date}")
                    print("=" * 40)
                    print(daily_df.to_string(index=False))
            
            elif choice == '2':
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)
                mask = (pd.to_datetime(df['Date']) >= start_date) & (pd.to_datetime(df['Date']) <= end_date)
                weekly_df = df[mask]
                if weekly_df.empty:
                    print("No records found for the past week")
                else:
                    print(f"\nWeekly Attendance Report ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
                    print("=" * 60)
                    print(weekly_df.to_string(index=False))
            
            elif choice == '3':
                month = input("Enter month (MM) or press Enter for current month: ")
                year = input("Enter year (YYYY) or press Enter for current year: ")
                
                if not month:
                    month = datetime.now().strftime("%m")
                if not year:
                    year = datetime.now().strftime("%Y")
                
                monthly_df = df[df['Date'].str.startswith(f"{year}-{month}")]
                if monthly_df.empty:
                    print(f"No records found for {year}-{month}")
                else:
                    print(f"\nMonthly Attendance Report for {year}-{month}")
                    print("=" * 50)
                    print(monthly_df.to_string(index=False))
            
            elif choice == '4':
                start_date = input("Enter start date (YYYY-MM-DD): ")
                end_date = input("Enter end date (YYYY-MM-DD): ")
                
                mask = (pd.to_datetime(df['Date']) >= pd.to_datetime(start_date)) & \
                       (pd.to_datetime(df['Date']) <= pd.to_datetime(end_date))
                custom_df = df[mask]
                
                if custom_df.empty:
                    print(f"No records found between {start_date} and {end_date}")
                else:
                    print(f"\nAttendance Report from {start_date} to {end_date}")
                    print("=" * 50)
                    print(custom_df.to_string(index=False))
            
            elif choice == '5':
                departments = set()
                for metadata in self.user_metadata.values():
                    if metadata.get('department'):
                        departments.add(metadata['department'])
                
                if not departments:
                    print("No department information found")
                    return
                
                print("\nAvailable Departments:")
                for i, dept in enumerate(departments, 1):
                    print(f"{i}. {dept}")
                
                dept_choice = input("\nEnter department number: ")
                try:
                    selected_dept = list(departments)[int(dept_choice) - 1]
                    dept_users = [name for name, metadata in self.user_metadata.items() 
                                if metadata.get('department') == selected_dept]
                    dept_df = df[df['Name'].isin(dept_users)]
                    
                    if dept_df.empty:
                        print(f"No records found for department: {selected_dept}")
                    else:
                        print(f"\nAttendance Report for Department: {selected_dept}")
                        print("=" * 50)
                        print(dept_df.to_string(index=False))
                except:
                    print("Invalid department selection")
            
            else:
                print("Invalid choice")
                
        except Exception as e:
            logging.error(f"Error generating report: {str(e)}")
            print("An error occurred while generating the report")

    def display_menu(self):
        while True:
            print("\nFace Recognition Attendance System")
            print("=================================")
            print("1. Register New Face")
            print("2. Take Attendance")
            print("3. Generate Report")
            print("4. Delete Face")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == '1':
                self.register_new_face()
            elif choice == '2':
                self.start_attendance()
            elif choice == '3':
                self.generate_report()
            elif choice == '4':
                self.delete_face()
            elif choice == '5':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    system = FaceAttendanceSystem()
    system.display_menu()
