from tensorflow.keras.models import load_model
import cv2
import numpy as np
from utils import sigmoid
from math import exp


class YoloDetector:
    """
    Represents an object detector for robot soccer based on the YOLO algorithm.
    """
    def __init__(self, model_name, anchor_box_ball=(5, 5), anchor_box_post=(2, 5)):
        """
        Constructs an object detector for robot soccer based on the YOLO algorithm.

        :param model_name: name of the neural network model which will be loaded.
        :type model_name: str.
        :param anchor_box_ball: dimensions of the anchor box used for the ball.
        :type anchor_box_ball: bidimensional tuple.
        :param anchor_box_post: dimensions of the anchor box used for the goal post.
        :type anchor_box_post: bidimensional tuple.
        """
        self.network = load_model(model_name + '.hdf5')
        self.network.summary()  # prints the neural network summary
        self.anchor_box_ball = anchor_box_ball
        self.anchor_box_post = anchor_box_post

    def detect(self, image):
        """
        Detects robot soccer's objects given the robot's camera image.

        :param image: image from the robot camera in 640x480 resolution and RGB color space.
        :type image: OpenCV's image.
        :return: (ball_detection, post1_detection, post2_detection), where each detection is given
                by a 5-dimensional tuple: (probability, x, y, width, height).
        :rtype: 3-dimensional tuple of 5-dimensional tuples.
        """
        # [DONE] Todo: implement object detection logic

        # Preprocess Image
        image = self.preprocess_image(image)

        # Use loaded model for prediction
        output = self.network.predict(image)

        # Process prediction
        ball_detection, post1_detection, post2_detection = self.process_yolo_output(output)

        return ball_detection, post1_detection, post2_detection

    def preprocess_image(self, image):
        """
        Preprocesses the camera image to adapt it to the neural network.

        :param image: image from the robot camera in 640x480 resolution and RGB color space.
        :type image: OpenCV's image.
        :return: image suitable for use in the neural network.
        :rtype: NumPy 4-dimensional array with dimensions (1, 120, 160, 3).
        """
        # [DONE] Todo: implement image preprocessing logic

        # Resize image
        image = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)

        # To NumPy
        image = np.array(image)

        # "Normalize" image
        image = image / 255.0

        # Reshape Image
        image = np.reshape(image, (1, 120, 160, 3))

        return image

    def process_yolo_output(self, output):
        """
        Processes the neural network's output to yield the detections.

        :param output: neural network's output.
        :type output: NumPy 4-dimensional array with dimensions (1, 15, 20, 10).
        :return: (ball_detection, post1_detection, post2_detection), where each detection is given
                by a 5-dimensional tuple: (probability, x, y, width, height).
        :rtype: 3-dimensional tuple of 5-dimensional tuples.
        """
        coord_scale = 4 * 8  # coordinate scale used for computing the x and y coordinates of the BB's center
        bb_scale = 640  # bounding box scale used for computing width and height
        output = np.reshape(output, (15, 20, 10))  # reshaping to remove the first dimension
        
        # [] Todo: implement YOLO logic

        # Get Biggest Probabilities
        biggest_pb = 0
        biggest_pb_idx = (None, None)
        biggest_pp = 0
        biggest_pp_idx = (None, None)
        biggest_pp2 = 0
        biggest_pp2_idx = (None, None)
        for i in range(15):
            for j in range(20):
                x = output[i][j]
                p_b = sigmoid(x[0])
                p_p = sigmoid(x[5])
                if p_b > biggest_pb:
                    biggest_pb = p_b
                    biggest_pb_idx = (i, j)
                if p_p > biggest_pp:
                    biggest_pp = p_p
                    biggest_pp_idx = (i, j)
                if p_p > biggest_pp2 and p_p < biggest_pp:
                    biggest_pp2 = p_p
                    biggest_pp2_idx = (i, j)

        # Extract features from x (for Ball)
        (ib, jb) = biggest_pb_idx
        x = output[ib][jb]
        tb, txb, tyb, twb, thb, tp, txp, typ, twp, thp = self.extract_features(x)
        
        # Transforms
        pb = sigmoid(tb)
        s_coord = 8 * 4
        (pwb, phb) = (5, 5)
        xb = (jb + sigmoid(txb)) * s_coord
        yb = (ib + sigmoid(tyb)) * s_coord
        wb = 640 * pwb * exp(twb)
        hb = 640 * phb * exp(thb)
        ball_detection = (pb, xb, yb, wb, hb)

        # Extract features from x (for Post1)
        (ip, jp) = biggest_pp_idx
        x = output[ip][jp]
        tb, txb, tyb, twb, thb, tp, txp, typ, twp, thp = self.extract_features(x)
        
        # Transforms
        pp = sigmoid(tp)
        s_coord = 8 * 4
        (pwp, php) = (2, 5)
        xp = (jp + sigmoid(txp)) * s_coord
        yp = (ip + sigmoid(typ)) * s_coord
        wp = 640 * pwp * exp(twp)
        hp = 640 * php * exp(thp)
        post1_detection = (pp, xp, yp, wp, hp)

        # Extract features from x (for Post2)
        (ip, jp) = biggest_pp2_idx
        x = output[ip][jp]
        tb, txb, tyb, twb, thb, tp, txp, typ, twp, thp = self.extract_features(x)
        
        # Transforms
        pp = sigmoid(tp)
        s_coord = 8 * 4
        (pwp, php) = (2, 5)
        xp = (jp + sigmoid(txp)) * s_coord
        yp = (ip + sigmoid(typ)) * s_coord
        wp = 640 * pwp * exp(twp)
        hp = 640 * php * exp(thp)
        post2_detection = (pp, xp, yp, wp, hp)

        return ball_detection, post1_detection, post2_detection

    def extract_features(self, x):
        return x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]
