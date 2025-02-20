# FaceGuard: AI-Powered Face Recognition Attendance System

FaceGuard is an advanced, AI-driven face recognition attendance system designed to streamline attendance management using cutting-edge computer vision techniques. Built with Python and OpenCV, FaceGuard leverages the LBPH (Local Binary Patterns Histograms) algorithm to accurately detect, register, and recognize faces in real time. This system is perfect for classrooms, offices, and events where efficient attendance recording is essential.

---

## Features

- **Robust Face Registration:** Capture and register faces using your webcam. The system collects multiple facial images to enhance recognition accuracy.
- **Real-Time Recognition:** Utilizes OpenCV's LBPH algorithm to perform fast and accurate facial recognition.
- **Automated Attendance Marking:** Automatically records attendance with date and time stamps into a CSV file.
- **Report Generation:** Easily generate detailed attendance reports for review.
- **User-Friendly Interface:** A simple console-based menu guides users through registration, attendance, and reporting.
- **Efficient Data Handling:** Uses Pandas for managing attendance data, ensuring that records are easily accessible and manageable.

---

## Technologies Used

- **Python:** The primary programming language.
- **OpenCV:** For real-time face detection and recognition.
- **NumPy:** For numerical and image processing operations.
- **Pandas:** For CSV-based attendance data management.
- **Pickle:** For serializing and deserializing the face recognition model and user data.

---

## Getting Started

### Prerequisites

- Python 3.7 or higher.
- A webcam (for live face capture and recognition).

### Installation

Clone the repository from GitHub:

```bash
git clone https://github.com/Sagexd08/FaceGuard.git
cd FaceGuard
```

Install the required dependencies:

```bash
pip install opencv-python opencv-contrib-python numpy pandas
```

### Running the Application

To start the face recognition attendance system, run:

```bash
python face_attendance_system.py
```

Follow the on-screen instructions to register faces, mark attendance, and generate reports.

---

## How It Works

1. **Register New Faces:**  
   When you choose to register a new face, the system captures multiple images from your webcam to build a robust dataset. Each image is saved and later used for training the recognition model.

2. **Training the Model:**  
   After face registration, the system automatically trains the LBPH face recognizer with the newly captured data. The trained model is then serialized for future use.

3. **Real-Time Attendance:**  
   The system activates the webcam to perform real-time face detection. When a registered face is recognized, attendance is marked automatically in a CSV file along with the current date and time.

4. **Report Generation:**  
   You can generate a detailed report of the attendance records at any time. The report is displayed in a tabular format for easy review.

---

## Future Improvements

- **Graphical User Interface (GUI):** Enhance the user experience with a dedicated GUI.
- **Email Notifications:** Add functionality to send automated email confirmations upon marking attendance.
- **Cloud Integration:** Store attendance data on the cloud for centralized access and backup.
- **Multi-Camera Support:** Allow integration with multiple cameras for larger-scale implementations.

---

## MIT License

```
MIT License

Copyright (c) 2025 Sagexd08

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Contributing

Contributions are always welcome! To contribute:

1. Fork the repository.
2. Create your feature branch:  
   ```bash
   git checkout -b feature/YourFeature
   ```
3. Commit your changes with clear messages.
4. Push to your branch:  
   ```bash
   git push origin feature/YourFeature
   ```
5. Open a Pull Request describing your changes.

---

## Contact

For any queries or feedback, please reach out to Sagexd08 at sohomchatterjee07@example.com](mailto:sohomchatterjee07@gmail.com).

---

## Acknowledgements

- Special thanks to the OpenCV community for their robust libraries and tutorials.
- Gratitude to the developers and contributors who make open-source projects possible.

---

Enjoy using FaceGuard for your face recognition attendance needs!
