import numpy as np
import math as m
import copy


def quantization_table(x=1):
    if (x == 1):
        Table = [[1, 1, 1, 1, 1, 2, 2, 4],
                 [1, 1, 1, 1, 1, 2, 2, 4],
                 [1, 1, 1, 1, 2, 2, 2, 4],
                 [1, 1, 1, 1, 2, 2, 4, 8],
                 [1, 1, 2, 2, 2, 2, 4, 8],
                 [2, 2, 2, 2, 2, 4, 8, 8],
                 [2, 2, 2, 4, 4, 8, 8, 16],
                 [4, 4, 4, 4, 8, 8, 16, 16]]
    elif (x == 2):
        Table = [[1, 2, 4, 8, 16, 32, 64, 128],
                 [2, 4, 4, 8, 16, 32, 64, 128],
                 [4, 4, 8, 16, 32, 64, 128, 128],
                 [8, 8, 16, 32, 64, 128, 128, 256],
                 [16, 16, 32, 64, 128, 128, 256, 256],
                 [32, 32, 64, 128, 128, 256, 256, 256],
                 [64, 64, 128, 128, 256, 256, 256, 256],
                 [128, 128, 128, 256, 256, 256, 256, 256]]
    else:
        raise ValueError("Error value in quantization table index")
    return Table


def DCT(block):
    output = np.zeros((8, 8))
    for u in range(8):
        for v in range(8):
            Basis = np.zeros((8, 8))
            for x in range(8):
                for y in range(8):
                    Basis[x, y] = np.cos((1 / 16) * (2 * x + 1) * u * m.pi) * np.cos((1 / 16) * (2 * y + 1) * v * m.pi)
            output[u, v] = sum(sum(block * Basis))
    return output

def normalize_DCT(output_of_dct):
    output_of_dct[:, 0] /= 2
    output_of_dct[0, :] /= 2
    output_of_dct /= 16
    return output_of_dct

def single_block_DCT(block):
    output = DCT(block)
    output = normalize_DCT(output)
    output = np.divide(output, quantization_table(1))
    output = np.round(output)
    output  = output.astype('int16')
    return output



def IDCT(block):  # To inverse the DCT Part and construct the 8*8 block before applying DCT
    Output = np.zeros((8, 8))
    for u in range(8):
        for v in range(8):
            Basis = np.zeros((8, 8))
            for x in range(8):
                for y in range(8):
                    Basis[x, y] = np.cos((1 / 16) * (2 * x + 1) * u * m.pi) * np.cos((1 / 16) * (2 * y + 1) * v * m.pi)
            Output = Output + block[u, v] * Basis
    return Output

def single_block_IDCT(block):
    output = np.multiply(block, quantization_table())
    output = IDCT(output)
    return output

def apply_on_all_sub_mats(input, func):
    output = np.zeros(input.shape)
    for anchor_row in range(0, input.shape[0], 8):
        for anchor_col in range(0, input.shape[1], 8):
            sub_matrix = input[anchor_row:anchor_row + 8, anchor_col:anchor_col + 8]
            output[anchor_row:anchor_row + 8, anchor_col:anchor_col + 8] = func(sub_matrix)
    return output


def dct_2d(input):
    return apply_on_all_sub_mats(input, single_block_DCT)

def idct_2d(input):
    return apply_on_all_sub_mats(input, single_block_IDCT)




####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################








