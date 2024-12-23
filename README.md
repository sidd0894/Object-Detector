# Real-Time Object Detection with EfficientDet Lite and OpenCV

This project demonstrates real-time object detection using a webcam feed, leveraging OpenCV and MediaPipe. The EfficientDet Lite0 model is used to detect objects, and bounding boxes are drawn around them, with class labels displayed on each frame.

## Requirements

Before running the project, ensure that the following Python libraries are installed:

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

Ensure the `efficientdet_lite0.tflite` model is placed in a `models` directory, as specified in the code, for object detection functionality.

## Functions

### `getObjects(frame, textColor=(0, 255, 0))`

This function processes an input frame and detects objects, drawing bounding boxes around each detected object.

**Parameters:**
- `frame` (numpy.ndarray): The image frame from the webcam feed, video, or image.
- `textColor` (tuple, optional): The color of the text (default is green).

**Returns:**
- A frame with bounding boxes drawn around detected objects.

### `main()`

This function continuously captures video from the webcam or loads an image/video file, applies object detection on each frame, and displays the result. The stream will stop when the user presses the 'q' key or reaches the end of the video.

## How It Works

1. **Object Detection**: The EfficientDet Lite0 model, loaded using MediaPipe's `ObjectDetector`, processes each frame to identify objects in real time.
2. **Bounding Boxes**: For each detected object, a bounding box is drawn using OpenCV’s `cv2.rectangle()`. The object's class label is displayed near the bounding box.
3. **Displaying Results**: The processed frame, now containing the bounding boxes and labels, is continuously displayed in a window.

## Running the Program

The program accepts command-line arguments to specify the input type (image, video, or webcam), input file, and output options. Below are the details for using these arguments:

### Arguments

- `--input-type` / (`-t`): **Required**  
  Specifies the type of input:
  - `image`: Use an image file as input.
  - `video`: Use a video file as input.
  - `webcam`: Use a webcam for live video input.

- `--index` / (`-i`):  
  Specifies the webcam index (only for webcam input). If you have multiple webcams, you can specify which one to use (default is `0`).

- `--input-file` / (`-if`):  
  Specifies the path to the input image or video file. This is required when `input-type` is either `image` or `video`.

- `--output-file` / (`-of`):  
  Specifies the path of the output file where the result will be saved. This can be an image file (for `image` input) or a video file (for `video` or `webcam` input). If not provided, the processed frames will only be displayed.

### Examples

#### For Webcam Input:

To start real-time object detection using the webcam, run:

```bash
python object_detection.py --input-type webcam
```

This will use the default webcam (`index 0`).  

You can use the `--index` / `-i` argument if you want to specify a specific webcam when the input type is set to `webcam`. For example:

  ```bash
  python object_detection.py --input-type webcam --index 1
  ```

  This will use the second webcam (if available) for object detection.  

#### For Image Input:

To run object detection on an image, provide the path to the image file using the `--input-file` / `-if` argument:

```bash
python object_detection.py --input-type image --input-file path_to_image.jpg --output-file output_image.jpg
```

This will detect objects in the specified image and save the processed image to `output_image.jpg`.

#### For Video Input:

To run object detection on a video file, provide the video file path using the `--input-file` / `-if` argument:

```bash
python object_detection.py --input-type video --input-file path_to_video.mp4 --output-file output_video.mp4
```

This will process the video and save the output with the detected objects to `output_video.mp4`.


## Note-

- If you choose to use a video or image file as input, make sure to specify the path to the input file using the `--input-file` / `-if` argument.

- The output file is optional. The processed image or video or webcam feed will be displayed live unless you specify an output file to save the result using `--output-file` / `-of`.

- The program uses `efficientdet_lite0.tflite`, which should be placed in a `models` directory to function correctly.

- Press 'q' to exit the program.