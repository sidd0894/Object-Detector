# Real-Time Object Detection with EfficientDet Lite and OpenCV

This project showcases real-time object detection from a webcam feed using OpenCV and MediaPipe. It employs the EfficientDet Lite object detection model to identify objects in frames captured from the webcam. The program draws bounding boxes around detected objects, providing an interactive view of the detection process.

## Requirements

Before running the project, ensure the following Python libraries are installed:

- `opencv-python`: For video capture and image processing.
- `mediapipe`: For vision tasks, including object detection.

To install the dependencies, run:

```bash
pip install mediapipe opencv-python
```

Alternatively, you can install all dependencies listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Ensure the `efficientdet_lite0.tflite` model is placed in a `models` directory to allow object detection functionality.

## Functions

### `getObjects(frame)`

This function processes an input frame and detects objects, drawing bounding boxes around each detected object.

**Parameters:**
- `frame` (numpy.ndarray): The image frame from the webcam feed.

**Returns:**
- A frame with bounding boxes drawn around detected objects.

### `main()`

This function continuously captures video from the webcam, applies object detection on each frame, and displays the result. The stream will stop when the user presses the 'q' key.

## How It Works

1. **Object Detection**: The EfficientDet Lite model, loaded using MediaPipe's `ObjectDetector`, processes each frame to identify objects in real time.
2. **Bounding Boxes**: For each detected object, a bounding box is drawn using OpenCV’s `cv2.rectangle()`. The object's class label is also displayed.
3. **Displaying Results**: The processed frame, now containing the bounding boxes and labels, is displayed continuously in a window.

## Running the Program

To launch the program, simply run the following command. It will start capturing video from your webcam and display the detected objects with bounding boxes.

```bash
python object_detection.py
```

To exit the program, press the 'q' key while the webcam feed is active.
