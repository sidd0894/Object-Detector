import cv2
import mediapipe as mp


model_path = 'models/efficientdet_lite0.tflite'

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    max_results=5,
    running_mode=VisionRunningMode.IMAGE)


def getObjects(frame):
    with ObjectDetector.create_from_options(options) as detector:

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(mp_image)

        if detection_result.detections:
            for detection in detection_result.detections:
                
                coords = detection.bounding_box
                x, y, w, h = (coords.origin_x, coords.origin_y, coords.width, coords.height)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, detection.categories[0].category_name, (x, y-2), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 0), 1)

        return frame


def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Unable to capture frame')
            break
        
        outputFrame = getObjects(frame)

        cv2.imshow('Frame', outputFrame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
if __name__ == '__main__':
    main()