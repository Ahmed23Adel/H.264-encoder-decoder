import numpy as np
import cv2
import math
from dct_2d import *
from source_coder import *
from huffman import *
def print_meta_data(path):
    vid_cap = cv2.VideoCapture(path)
    fps = vid_cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('[INFO] fps = ' + str(fps))
    print('[INFO] number of frames = ' + str(frame_count))
    success, image = vid_cap.read()
    print("[INFO] image shape {}".format(image.shape))
    return  cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).shape, fps


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


def get_best_motion_from(prev_frame, current_macro_block,block_size, anchor_row, anchor_col,search_x, search_y):
    least_mse = float('inf')
    motion_vector = [0, 0]  # move only anchor
    for s_row in range(max(0, anchor_row - search_y),
                       min(prev_frame.shape[0], anchor_row + block_size[0] + search_y)):  # on rows
        for s_col in range(max(0, anchor_col - search_x),
                           min(prev_frame.shape[1], anchor_col + block_size[1] + search_y)):  # on columns
            prev_frame_block = prev_frame[s_row:s_row + block_size[0], s_col:s_col + block_size[1]]
            if s_row + block_size[0] > prev_frame.shape[0] or s_col + block_size[1] > prev_frame.shape[1]:  # if it hits the end
                continue
            mse = ((current_macro_block - prev_frame_block) ** 2).mean(axis=None)
            if mse < least_mse:
                least_mse = mse
                motion_vector[0], motion_vector[1] = s_row - anchor_row, s_col - anchor_col
    return tuple(motion_vector)

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
            all_motion_estimation[idx, 0], all_motion_estimation[idx, 1] = \
                get_best_motion_from(prev_frame, current_macro_block,block_size, anchor_row, anchor_col,search_x, search_y)
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



def encode_H264(path, quan_type =1,test = True):
    """
    :param path: for the video to encode
    :param test: if true it will apply on the same procedure on simple matrix
    :return: encoded video
    """
    if test:
        return test_simple(path)
    block_size = (8,8)
    print("[INFO] reading {}".format(path))
    shape_img, fps = print_meta_data(path)
    prev_frame = np.zeros(shape_img)
    all_frames_reordered = [shape_img[0], shape_img[1], round(fps), block_size[0], block_size[1]]
    index  = 1
    for current_frame in  read_video(path):
        cv2.imwrite("{}prev_frame -- original.jpeg".format(index), prev_frame)
        cv2.imwrite("{}st frame -- original.jpeg".format(index), current_frame)
        all_motion_estimations = motion_estimation(current_frame,prev_frame,block_size)
        print(prev_frame.shape, all_motion_estimations.shape)
        # img_reord = reorder(prev_frame, all_motion_estimations)
        # all_frames_reordered.extend(img_reord)
        print("done all_motion_estimations")
        print(all_motion_estimations)
        predicted_frame  = motion_compensation(prev_frame,all_motion_estimations, block_size)
        print("done predicted_frame")
        cv2.imwrite("{}st frame -- predicted_frame.jpeg".format(index), predicted_frame)
        diff = current_frame - predicted_frame
        print("done diff")
        cv2.imwrite("{}st frame -- diff.jpeg".format(index), diff)
        diff_dct = dct_2d(diff)
        print("done diff_dct")
        cv2.imwrite("{}st frame -- diff_dct.jpeg".format(index), diff_dct)
        diff_idct = idct_2d(diff_dct)
        print("done diff_idct")
        cv2.imwrite("{}st frame -- diff_idct.jpeg".format(index), diff_idct)
        img_reord = reorder(diff_dct,all_motion_estimations)
        all_frames_reordered.extend(img_reord)
        print("done reorder")
        # entropy encoder
        index +=1
        prev_frame = predicted_frame + diff_idct
        if index ==5:
            break
    # Now all_frames_reordered contains all frames, and all_motion_estimations
    # Now we need to run Huffman Coding
    video_coded, video_dict = encode_huffman(all_frames_reordered)
    #np.savetxt('VideoData.csv', np.array(video_coded), delimiter=',')
    print("video_dict",video_dict)
    return  video_dict, video_coded


def get_frames_from_decoded_video(video_coded, video_dict ):
    shape_img, fps, block_size, video_decoded = get_video_meta(video_coded, video_dict)
    length_frame = int(shape_img[0] * shape_img[1]) + int(shape_img[0]/block_size[0] * shape_img[1]/block_size[1])*2 # Diff dims + motion vector length
    video_decoded = video_decoded[5:]
    for i in range(0, len(video_decoded), length_frame):
        yield video_decoded[i:i+length_frame]

def get_motion_vec_and_code(code, shape_img, block_size):
    diff_frame = int(shape_img[0] * shape_img[1])
    return code[:diff_frame], code[diff_frame:]

def get_video_meta(video_coded, video_dict):
    video_decoded = decode_huffman(video_coded, video_dict)
    video_decoded = dis_run_length(video_decoded)
    shape_img = [0, 0]
    block_size = [0, 0]
    shape_img[0], shape_img[1], fps, block_size[0], block_size[1] = video_decoded[:5]
    return (shape_img[0], shape_img[1]) , fps, (block_size[0], block_size[1]), video_decoded

def organize_motion_vector_code(motion_vector,motion_vectors_length):
    output_motion_vector = np.zeros((motion_vectors_length,2))
    idx = 0
    for i in range(0, len(motion_vector),2):
        output_motion_vector[idx ,0],output_motion_vector[idx ,1] = motion_vector[i:i+2]
        idx += 1
    return output_motion_vector
def decode_H264(video_coded, video_dict ):
    video_dict = dict((v, k) for k, v in video_dict.items())
    shape_img, fps, block_size, _ = get_video_meta(video_coded, video_dict)
    print(shape_img, fps, block_size)
    motion_vectors_length = int(shape_img[0]/block_size[0] * shape_img[1]/block_size[1])*2
    prev_frame = np.zeros((shape_img[0], shape_img[1]))
    index = 1
    for frame in get_frames_from_decoded_video(video_coded, video_dict):
        diff_code_1d, motion_vector_code_1d = get_motion_vec_and_code(frame, shape_img, block_size) # split based on image size, and block size
        print("Done diff_code_1d, motion_vector_code_1d ")
        motion_vector = organize_motion_vector_code(motion_vector_code_1d,motion_vectors_length)
        print("Done organize_motion_vector_code")
        predicted_frame = motion_compensation(prev_frame, motion_vector, block_size)
        print("Done prediction_frame")
        cv2.imwrite("Decoder, {}st frame -- predicted_frame.jpeg".format(index), predicted_frame)
        diff_code_2d = dis_reorder(diff_code_1d, block_size,shape_img)
        print("Done diff_code_2d")
        cv2.imwrite("Decoder, {}st frame -- diff_code_2d.jpeg".format(index), diff_code_2d)
        diff_idct = idct_2d(diff_code_2d)
        print("Done diff_idct")
        frame_reconstructed = predicted_frame + diff_idct # Current frame
        print("Done frame_reconstructed")
        cv2.imwrite("Decoder, {}st frame -- frame_reconstructed.jpeg".format(index), frame_reconstructed)
        prev_frame = frame_reconstructed
        index = index + 1



def test_simple(path):
    all_frames_reordered = []

    index = "simple"
    shape_img, fps = (256,256), 24
    prev_frame = np.zeros(shape_img)
    block_size = (8, 8)
    current_frame = cv2.imread(r"D:\Zewail\Year 4\info\project\H.264\H.264-encoder-decoder\camera_man.png", 0)
    cv2.imwrite("{}prev_frame -- original.jpeg".format(index), prev_frame)
    cv2.imwrite("{}st frame -- original.jpeg".format(index), current_frame)
    all_motion_estimations = motion_estimation(current_frame, prev_frame, block_size)
    all_frames_reordered = [shape_img[0], shape_img[1], round(fps), block_size[0], block_size[1]]
    print(prev_frame.shape, all_motion_estimations.shape)
    # img_reord = reorder(prev_frame, all_motion_estimations)
    # all_frames_reordered.extend(img_reord)
    print("done all_motion_estimations")
    print(all_motion_estimations)
    predicted_frame = motion_compensation(prev_frame, all_motion_estimations, block_size)
    print("done predicted_frame")
    cv2.imwrite("{}st frame -- predicted_frame.jpeg".format(index), predicted_frame)
    diff = current_frame - predicted_frame
    print("done diff")
    cv2.imwrite("{}st frame -- diff.jpeg".format(index), diff)
    diff_dct = dct_2d(diff)
    print("done diff_dct")
    cv2.imwrite("{}st frame -- diff_dct.jpeg".format(index), diff_dct)
    diff_idct = idct_2d(diff_dct)
    print("done diff_idct")
    cv2.imwrite("{}st frame -- diff_idct.jpeg".format(index), diff_idct)
    img_reord = reorder(diff_dct, all_motion_estimations)
    all_frames_reordered.extend(img_reord)
    print("done reorder")
    # entropy encoder
    prev_frame = predicted_frame + diff_idct
    cv2.imwrite("{}st frame -- prev_frame.jpeg".format(index), prev_frame)
    video_coded, video_dict  = encode_huffman(all_frames_reordered)
    # np.savetxt('VideoData.csv', np.array(video_coded), delimiter=',')
    print("video_dict", video_dict)
    return video_dict, video_coded
