# Facial Detection App

Simple real-time facial detection and privacy app utilizing cv2 Cascade Classifier CNN. Detects multiple face in a live video capture from your computer.

Bounding boxes and coordinates for detected faces are displayed on the live frame and faces are pixelates for privacy. You can disable pixelation by setting the `pixelate` variable in [./src/main.py](./src/main.py) to `False`.

Setup
=====

Run `conda env create -f environment.yml` for local development Python environment.

Run [./src/main.py](./src/main.py) to start application.