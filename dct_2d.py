import numpy as np
import math
import copy


def quantization(x):
    if x == 1:
        return [[1, 1, 1, 1, 1, 2, 2, 4],
                 [1, 1, 1, 1, 1, 2, 2, 4],
                 [1, 1, 1, 1, 2, 2, 2, 4],
                 [1, 1, 1, 1, 2, 2, 4, 8],
                 [1, 1, 2, 2, 2, 2, 4, 8],
                 [2, 2, 2, 2, 2, 4, 8, 8],
                 [2, 2, 2, 4, 4, 8, 8, 16],
                 [4, 4, 4, 4, 8, 8, 16, 16]]
    elif x == 2:
        return [[1, 2, 4, 8, 16, 32, 64, 128],
                 [2, 4, 4, 8, 16, 32, 64, 128],
                 [4, 4, 8, 16, 32, 64, 128, 128],
                 [8, 8, 16, 32, 64, 128, 128, 256],
                 [16, 16, 32, 64, 128, 128, 256, 256],
                 [32, 32, 64, 128, 128, 256, 256, 256],
                 [64, 64, 128, 128, 256, 256, 256, 256],
                 [128, 128, 128, 256, 256, 256, 256, 256]]


def single_block_DCT_(block):
    """

    :param x_in: should be 8x8 matrix
    :return: dct of x_in
    """
    assert block.shape == (8,8)
    output = np.zeros((8, 8))
    for u in range(8):
        for v in range(8):
            basis = np.zeros((8, 8))
            for x in range(8):
                for y in range(8):
                    basis[x, y] = np.cos((1 / 16) * (2 * x + 1) * u * math.pi) * np.cos((1 / 16) * (2 * y + 1) * v * math.pi)
            output[u, v] = sum(sum(block * basis))
    return output




def normalize(dct_output):
    """
    :param dct_output: output of DCT that need to be normalized
    :return: normalized matrix
    """
    dct_output_new =  copy.deepcopy(dct_output)
    dct_output_new[0,:] /= 2
    dct_output_new[:,0] /= 2
    dct_output_new /= 16
    return dct_output_new


def single_block_DCT(x_in):
    output = single_block_DCT_(x_in)
    output = normalize(output)
    output = np.divide(output, quantization(1)).astype(int)
    return output

def single_block_IDCT(block):
    output = np.zeros((8, 8))
    for u in range(8):
        for v in range(8):
            basis = np.zeros((8, 8))
            for x in range(8):
                for y in range(8):
                    basis[x, y] = np.cos((1 / 16) * (2 * x + 1) * u * math.pi) * np.cos((1 / 16) * (2 * y + 1) * v * math.pi)
            output = output + block[u, v] * basis
    output = np.multiply(output, quantization(1))
    return output

def dct_2d(input):
    output = np.zeros(input.shape)
    for anchor_row in range(0, input.shape[0], 8):
        for anchor_col in range(0, input.shape[1], 8):
            sub_matrix = input[anchor_row:anchor_row + 8, anchor_col:anchor_col + 8]
            output[anchor_row:anchor_row + 8, anchor_col:anchor_col + 8] = single_block_DCT(sub_matrix)
    return output

def idct_2d(input):
    output = np.zeros(input.shape)
    for anchor_row in range(0, input.shape[0], 8):
        for anchor_col in range(0, input.shape[1], 8):
            sub_matrix = input[anchor_row:anchor_row + 8, anchor_col:anchor_col + 8]
            output[anchor_row:anchor_row + 8, anchor_col:anchor_col + 8] = single_block_IDCT(sub_matrix)
    return output








