# Face Attendance System
A simple application to recognise faces and check for attendance.

## Setup
The project can be setup using [uv](https://github.com/astral-sh/uv) or
traditionally with pip.

#### With pip:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For uv, installation of deps will happens automatically with the first run.

## Usage
#### With uv
```bash
uv run app.py
```

#### With pip
Ensure the `.venv` is activated.
```bash
python app.py
```

Press <kbd>n</kbd> to capture a new face and enter a name in the terminal.
The application should then recognise the face by displaying the name in the
top left corner.

