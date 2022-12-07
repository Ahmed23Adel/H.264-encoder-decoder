from encoder_2d import *


if __name__ == "__main__":
    video_dict, video_decoded = encode_H264(r"D:\Zewail\Year 4\info\project\H.264\H.264-encoder-decoder\video_sample_cut.mp4", test=False)
    # decode_H264(video_decoded, video_dict)