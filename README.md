# Face Recognition Attendance System 📸✅

A real-time face recognition attendance system using OpenCV and dlib. Register faces once — the system detects and logs attendance automatically at 15–25 FPS.

---

## Demo

| Action | How |
|---|---|
| Register a user | Press `N` key while camera is running |
| Mark attendance | System auto-detects and logs recognized faces |
| View logs | Check the generated attendance CSV |

---

## Features

- 🎥 **Real-time face detection** at 15–25 FPS via OpenCV
- 🧠 **High-accuracy recognition** using dlib's 128-dimension face encodings
- ⚡ **Fast startup** — pre-serialized encodings loaded once at launch (not recomputed per frame)
- 👥 **Supports up to 50 registered users**
- 📝 **Automatic attendance logging** with timestamps
- 🪶 **Lightweight** — runs on standard hardware, no GPU required
- 🔧 **Easy setup** using virtualenv

---

## Tech Stack

| Component | Technology |
|---|---|
| Face Detection | OpenCV |
| Face Recognition | face_recognition (dlib) |
| Encoding Serialization | Pickle |
| Environment | Python virtualenv |
| Language | Python 3.8+ |

---

## Project Structure

```
face-attendance-system/
├── main.py                # Main script — camera loop + recognition
├── encode_faces.py        # One-time encoding generator for registered users
├── known_faces/           # Folder containing registered user images
│   └── [name]/            # One subfolder per person
│       └── photo.jpg
├── encodings.pkl          # Pre-serialized face encodings (auto-generated)
├── attendance.csv         # Output attendance log with timestamps
├── requirements.txt
└── README.md
```

---

## Setup & Run

### 1. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** dlib requires CMake. Install it first:
> - Windows: `pip install cmake` then `pip install dlib`
> - Linux: `sudo apt install cmake libopenblas-dev` then `pip install dlib`

### 3. Register faces

Add a folder for each person inside `known_faces/`:
```
known_faces/
└── Rajnish/
    └── rajnish.jpg
```
Then generate encodings:
```bash
python encode_faces.py
```

### 4. Run the system

```bash
python main.py
```

- Press `N` to register a new user live from the camera
- Press `Q` to quit

---

## How It Works

```
Camera Frame → OpenCV (detect face locations)
    → dlib (compute 128-D face encoding)
    → Compare against known encodings (Euclidean distance)
    → Match found → Log name + timestamp to attendance.csv
    → Draw bounding box + name on frame
```

---

## Performance

| Metric | Value |
|---|---|
| Frame Rate | 15–25 FPS |
| Max Users | 50 |
| Recognition Accuracy | ~99% (well-lit, frontal faces) |
| Startup Time | < 2 seconds (pre-serialized encodings) |

---

## Limitations

- Works best with good lighting and frontal faces
- Recognition accuracy drops with masks or heavy occlusion
- Designed for single-camera setups

---

## Future Scope

- Multi-camera support
- Web dashboard for attendance reports
- Cloud sync for enterprise deployments

---

## Author

**Rajnish Kumar** — [github.com/Rajnish7569](https://github.com/Rajnish7569) | [LinkedIn](https://linkedin.com/in/rajnish7569)
