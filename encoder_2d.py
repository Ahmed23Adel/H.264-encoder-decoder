import numpy as np
import cv2
import math
def print_meta_data(path):
    vid_cap = cv2.VideoCapture(path)
    fps = vid_cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('[INFO] fps = ' + str(fps))
    print('[INFO] number of frames = ' + str(frame_count))
    success, image = vid_cap.read()
    print("[INFO] image shape {}".format(image.shape))
    return  cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).shape


def read_video(path):
    """

    :param path: for the video to encode
    :yield each frame.
    I didin't use return, as that is more memory efficient
    """
    vid_cap = cv2.VideoCapture(path)
    success, image = vid_cap.read()
    success = True
    while success:
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        yield gray_img
        success, image = vid_cap.read()




def motion_estimation(prev_frame, current_frame,block_size, search_x = 16, search_y=16):
    """
    :param prev_frame: Preview frame to get the prediction from it
    :param current_frame: current frame to encode
    :param block_size: block size to be used in dividing the image into multiples of it.
    :param search_x: search areay in the x-axis
    :param search_y: search areay in the y-axis
    :return: all motion estimataoin start from all blocks in first row then going down
    """
    all_motion_estimation = np.zeros((int(current_frame.shape[0]/block_size[0] * current_frame.shape[1]/block_size[1]), 2))
    idx = 0
    for anchor_row in range(0,prev_frame.shape[0], block_size[0]): # loop on all rows
        for anchor_col in range(0,prev_frame.shape[1], block_size[1]):# loop on all cols
            current_macro_block = current_frame[anchor_row:anchor_row+block_size[0], anchor_col:anchor_col+block_size[1]]
            # search_x and search_y
            least_mse = float('inf')
            motion_vector = [0,0] # move only anchor
            for s_row in range(max(0,anchor_row-search_y), min(current_frame.shape[0],anchor_row+block_size[0]+search_y) ): # on rows
                for s_col in range(max(0,anchor_col-search_x), min(current_frame.shape[1],anchor_col+block_size[1]+search_x)): # on columns
                    prev_frame_block  = prev_frame[s_row:s_row+block_size[0], s_col:s_col+block_size[1]]
                    if s_row+block_size[0] > current_frame.shape[0] or s_col+block_size[1] > current_frame.shape[1]: # if it hits the end
                        continue
                    mse = ((current_macro_block - prev_frame_block) ** 2).mean(axis=None)
                    if mse < least_mse:
                        least_mse = mse
                        motion_vector[0], motion_vector[1] =s_row - anchor_row ,   s_col - anchor_col
            all_motion_estimation[idx, 0] = motion_vector[0]
            all_motion_estimation[idx, 1] = motion_vector[1]
            idx = idx + 1
    return all_motion_estimation


def motion_compensation(prev_frame,all_motion_estimations, block_size):
    """
    :param prev_frame: Preview frame to get the prediction from it
    :param all_motion_estimations: what should be returned from motion_estimation
    :param block_size: block size to be used in dividing the image into multiples of it.
    :return: predicted image
    """
    predicted_frame = np.zeros(prev_frame.shape)
    idx = 0
    for anchor_row in range(0, prev_frame.shape[0], block_size[0]):
        for anchor_col in range(0, prev_frame.shape[1], block_size[1]):
            new_anchor_row = int(anchor_row + all_motion_estimations[idx,0])
            new_anchor_col = int(anchor_col + all_motion_estimations[idx, 1])
            predicted_frame[anchor_row:anchor_row+block_size[0], anchor_col:anchor_col+block_size[1]] = prev_frame[new_anchor_row:new_anchor_row+block_size[0], new_anchor_col:new_anchor_col+block_size[1]]
            idx = idx + 1

    return predicted_frame



def encode_H264(path, test = True):
    """
    :param path: for the video to encode
    :param test: if true it will apply on the same procedure on simple matrix
    :return: encoded video
    """
    if test:
        return test_simple(path)
    block_size = (16,16)
    print("[INFO] reading {}".format(path))
    shape_img = print_meta_data(path)
    prev_frame = np.zeros(shape_img)
    for current_frame in  read_video(path):
        all_motion_estimations = motion_estimation(current_frame,prev_frame,block_size)
        prev_frame = current_frame
        break



def test_simple(path):
    c_f = np.array([[1] * 16 + [2] * 16] * 16 + [[3] * 16 + [4] * 16] * 16)
    p_f = np.array([[2] * 16 + [1] * 16] * 16 + [[4] * 16 + [3] * 16] * 16)
    c_f[5, 5] = 10
    c_f[18, 5] = 20
    c_f[17, 28] = 20
    c_f[28, 28] = 20
    all_motion_estimations = motion_estimation(c_f, p_f, (16, 16))
    for v in all_motion_estimations:
        print(v)
    p = motion_compensation(p_f, all_motion_estimations, (16, 16))
    print(p)
    for x in p:
        print(x)