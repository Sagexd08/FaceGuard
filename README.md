# FaceGuard: AI-Powered Face Recognition Attendance System

FaceGuard is an advanced, AI-driven face recognition attendance system designed to streamline attendance management using cutting-edge computer vision techniques. Built with Python and OpenCV, FaceGuard leverages the LBPH (Local Binary Patterns Histograms) algorithm to accurately detect, register, and recognize faces in real time. Ideal for classrooms, offices, and events, FaceGuard offers an efficient solution for managing attendance while ensuring data integrity and user convenience.

---

## Key Enhancements in This Version

- **Quality Checks for Face Registration:**  
  Implements automatic quality assessments (including brightness and contrast metrics) during face capture to ensure only high-quality images are used for registration.

- **Automatic Backups with Cleanup & Multi-Threaded Backup System:**  
  Regularly backs up attendance and registration data with an automated cleanup process, optimized by a multi-threaded backup system to maintain performance and data integrity.

- **Comprehensive Reporting Options:**  
  Provides detailed, customizable reports that offer deeper insights into attendance patterns, supporting advanced filtering and analysis.

- **Department-Based Organization:**  
  Organizes users and attendance records by department, making it easier to manage and review data in large-scale implementations.

- **User Metadata Storage:**  
  Stores additional metadata for each user (such as department, role, and contact information) to enhance record keeping and reporting.

- **Better Error Handling and Logging:**  
  Features improved error management with robust logging mechanisms, making troubleshooting and system maintenance more efficient.

- **Progress Visualization During Registration:**  
  Offers real-time visual feedback during the face registration process, ensuring users are informed about the capture progress.

- **Confirmation Steps for Deletions:**  
  Adds safety confirmation prompts before any deletion operation, reducing the risk of accidental data loss.

---

## Features

- **Robust Face Registration:**  
  Capture and register faces using your webcam. The system now includes quality checks that assess face quality based on brightness and contrast. A progress visualization indicator guides the user through the multi-step capture process.

- **Real-Time Recognition:**  
  Leverages OpenCV’s LBPH algorithm to perform fast, accurate face detection and recognition in real time.

- **Automated Attendance Marking:**  
  Automatically records attendance with date and time stamps, and now organizes records by department for streamlined data management.

- **Enhanced Data Management & Backups:**  
  - **Automatic Backups:** Regular, multi-threaded backups ensure data safety, with an automatic cleanup routine to manage storage efficiently.  
  - **User Metadata Storage:** Maintain detailed profiles for each user to support comprehensive attendance analysis.

- **Report Generation:**  
  Easily generate detailed, customizable attendance reports. The new comprehensive reporting options allow for advanced data analysis and filtering.

- **User-Friendly Interface:**  
  A simple console-based menu guides users through registration, attendance marking, and reporting, now with added confirmation steps for sensitive operations such as deletions.

- **Efficient Data Handling:**  
  Uses Pandas for managing attendance data, ensuring that records are easily accessible and manageable, while improved error handling and logging facilitate maintenance.

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
   The system captures multiple images from your webcam with built-in quality checks (evaluating metrics like brightness and contrast) and progress visualization to ensure a robust dataset is created. Each image is saved and later used for training the recognition model.

2. **Training the Model:**  
   After face registration, the system automatically trains the LBPH face recognizer using the newly captured data. The trained model is then serialized for future use.

3. **Real-Time Attendance:**  
   The system activates the webcam for real-time face detection. When a registered face is recognized, attendance is marked automatically in a CSV file with the current date and time, organized by department.

4. **Data Management & Backup:**  
   Regular backups are performed using a multi-threaded system with automatic cleanup, ensuring your data is secure and storage remains optimized. User metadata is stored alongside attendance records for detailed reporting.

5. **Report Generation:**  
   Generate detailed attendance reports at any time. The comprehensive reporting options display data in a tabular format with advanced filtering capabilities for easy review.

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

For any queries or feedback, please reach out to Sagexd08 at [sohomchatterjee07@gmail.com](mailto:sohomchatterjee07@gmail.com).

---

## Acknowledgements

- Special thanks to the OpenCV community for their robust libraries and tutorials.
- Gratitude to the developers and contributors who make open-source projects possible.

---

Enjoy using FaceGuard for your face recognition attendance needs!

---

