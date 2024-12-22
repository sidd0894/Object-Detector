import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print('Starting...\n')

import cv2
import mediapipe as mp
import argparse


model_path = 'models/efficientdet_lite0.tflite'

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    max_results=5,
    running_mode=VisionRunningMode.IMAGE)


def getObjects(frame, textColor=(0, 255, 0)):
    with ObjectDetector.create_from_options(options) as detector:

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(mp_image)

        if detection_result.detections:
            for detection in detection_result.detections:
                coords = detection.bounding_box
                x, y, w, h = (coords.origin_x, coords.origin_y, coords.width, coords.height)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, detection.categories[0].category_name, (x, y-4), cv2.FONT_HERSHEY_COMPLEX, 0.8, textColor, 1)

        return frame


def main():
    filename = os.path.basename(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description='description: Perform real-time object detection with EfficientDet Lite0 on images, videos, or webcam input.',
        usage= f"'python {filename} [-h]' for list of usable commands."
    )

    parser.add_argument('--input-type', '-t', choices=['image', 'video', 'webcam'], required=True, help='Select input type: image, video or webcam')
    parser.add_argument('--index', '-i', default=0, type=int, help='Webcam index for input type \'webcam\' (default: 0)', metavar='')
    parser.add_argument('--input-file', '-if', type=str, help='Path of input image or video file (optional)', metavar='')
    parser.add_argument('--output-file', '-of', type=str, help='Path of output file to save the output image or video (.mp4/.avi)  or webcam video (.mp4/.avi) (optional)', metavar='')
    args = parser.parse_args()


    if args.input_type in ['image', 'video']:
        if not args.input_file:
            parser.error(f"'--input-file/-if' is required when input type is '{args.input_type}'")

        elif args.index:
            parser.error(f"'--index/-i' should not be used when input type is '{args.input_type}'")

    elif args.input_type == 'webcam' and args.input_file:
        parser.error("'--input-file/-if' should not be used when input type is 'webcam'")


    if args.input_type == 'image':
        img = cv2.imread(args.input_file)
        output_img = getObjects(img)

        cv2.imwrite(args.output_file, output_img) if args.output_file else None
        cv2.imshow('Output Image', output_img)
        cv2.waitKey(0)


    else:
        cap = cv2.VideoCapture(args.index if args.input_type == 'webcam' else args.input_file)

        if not cap.isOpened():
            print('[ERROR] Unable to open the video source')
            exit()
        
        fourcc = ''
        out = ''
        frameW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if args.output_file:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(args.output_file, fourcc, float(fps), (frameW, frameH))

        while True:
            ret, frame = cap.read()
            if not ret:
                print('[ERROR] Unable to capture frames')
                break
            
            outputFrame = getObjects(frame)

            try:
                out.write(outputFrame) if args.output_file else None

            except:
                print(f'[ERROR] Unable to write output to {args.output_file}')
                break

            cv2.imshow('Frame', outputFrame)
            # if cv2.waitKey(1 if args.input_type == 'webcam' else int(1000/fps)) & 0xFF == ord('q'):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release() if args.output_file else None

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()