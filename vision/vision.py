import cv2 as cv
import numpy as np
import tensorflow as tf

class DetectedObject:
    def __init__(self, label, frameLocation) -> None:
        self.label = label
        self.frameLoaction = frameLocation

class Vision:
    def __init__(self, weights):
        # Load the weights (these are the weight parameters that define the model)
        self.model = tf.saved_model.load(weights)
        # Still need to figure out what this is doing explicitly. I know that the signatures are related to how the model itself 
        self.infer = self.model.signatures["serving_default"]

    # Using the provided source and model, return any detected objects in the frame
    def detect(self, source):
        try:
            #--- Preprocess frame ---#
            # Convert from BRG to RGB color space (not sure why, though it likely has to do with how tensorflow expects its input to be formatted)
            # Will this work with BGR2GRAY?
            frame_data = cv.cvtColor(source, cv.COLOR_BGR2RGB)
            # Resize the input frame to match the size of images that the model was trained on
            # TODO: Change these values to vriables (can we extract the size of the training data from the saved model?)
            frame_data = cv.resize(frame_data, (416, 416))

            frame_data = frame_data / float(255)

            # This converts our image data to a tensorlike object (an np.array) so that the image data can be converted into a tensor
            # See `tf.constant()`` implementation in the `Make Predictions for ROIs` section
            frame_data = frame_data[np.newaxis, ...].astype(np.float32)

        except:
            # TODO: Implement better error handling
            print("There was an error preprocessing the source frame")
            return

        #--- Make Predictions for ROIs ---#
        # Create a tensor object from the image data (More info on the process can be found here: https://www.tensorflow.org/api_docs/python/tf/constant)
        batch_data = tf.constant(frame_data)
        # Use the signatures of the the trained model to make predictions about potential ROIs
        # This returns a ??? object
        predictions = self.infer(batch_data)

        # Get all ROIs (represented as a 4D array) and corresponding confidence levels from the predictions
        for key, value in predictions.items():
            boxes = value[:, :, 0:4]
            confidence_levels = value[:, :, 4:]

        #--- Clean Prediction Data ---#
        # The `combined_non_max_suppression` method is an optimzation function that cleans the prediction data
        # i.e. it removes ROIs that overlap by a certain threshold value as well as removes ROIs whose predicted cofidence scores are below a threshold value
        # The function returns four tensor objects that represent the non-max suppressed ROI boxes, the cofidence scores for said boxes, the classification for each box, and the valid dectections for the batch_data (i.e. the top detection entries in the preceeding tensors)
        # More info here: https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression
        boxes, scores, classes, detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(confidence_levels, (tf.shape(confidence_levels)[0], -1, tf.shape(confidence_levels)[-1])),
            max_output_size_per_class=5,
            max_total_size=5,
            iou_threshold=0.25,
            score_threshold=0.50,
        )

        #--- Draw Bounding Boxes for ROIs ---#
        # print(boxes, scores, classes, detections)

        # print('\n----------\n')

        output_image = source

        def read_class_names(class_file_name):
            names = {}
            with open(class_file_name, 'r') as data:
                for ID, name in enumerate(data):
                    names[ID] = name.strip('\n')
            return names

        classDict = read_class_names("./data/classes/coco.names")

        # Replace with new util function that will handle  drawing
        for i in range(detections[0]):
            font = cv.FONT_HERSHEY_SIMPLEX

            print(boxes[0][i])
            tly = int(boxes[0][i][0] * 480)
            bry = int(boxes[0][i][2] * 480)
            tlx = int(boxes[0][i][1] * 640)
            brx = int(boxes[0][i][3] * 640)

            print(tlx, tly, brx, bry)

            print(classDict[int(classes[0][i])])
            print(tf.get_static_value(scores[0][i]))

            score = round(tf.get_static_value(scores[0][i]), 3)

            cv.rectangle(output_image, (tlx, tly), (brx, bry), (0, 0, 255), 2)
            cv.putText(output_image, str(classDict[int(classes[0][i])]), (tlx,tly - 25), font, 0.5, (0, 0, 255), 2, cv.LINE_AA)
            cv.putText(output_image, str(score), (tlx,tly - 10), font, 0.5,(0, 0, 255), 2, cv.LINE_AA)




        # Run detection algo

        # Return results
        cv.imshow("Detection", source)

    def orient(self):
        
        pass


### ### TESTING ### ###

from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())

vision = Vision("../../tensorflow-yolov4-tflite/checkpoints/yolov4-416")

capture = cv.VideoCapture(0)

while True:
    ret, frame = capture.read()

    vision.detect(frame)

    if cv.waitKey(1) == ord("q"):
        break

# ret, frame = capture.read()

# vision.detect(frame)

# cv.waitKey(0)

### ### ### ### ### ###