import numpy as np

def reorder(matrix_2d, motion_estimation_vectors):
    output_reord1 = []
    output_reord2 = []
    index = 0
    for anchor_row in range(0, matrix_2d.shape[0], 8):
        for anchor_col in range(0, matrix_2d.shape[1], 8):
            sub_matrix = matrix_2d[anchor_row:anchor_row + 8, anchor_col:anchor_col + 8]
            zig_read =  zig_zag_read(sub_matrix)
            output_sub_mat = run_length_encoder(zig_read)
            output_reord1.extend(output_sub_mat.astype(int))
            output_reord2.extend(run_length_encoder(motion_estimation_vectors[index]).astype(int))
            index += 1
    output_reord1.extend(output_reord2)
    return output_reord1

def zig_zag_read(block_2d):
    ## Just converting the 8*8 block into one array
    output_1d = []
    Counter = 0
    Condition = True
    for i in range(9):
        for j in range(i):
            if (Condition):
                if (Counter % 2 == 0):
                    output_1d.append(block_2d[j][i - j - 1])
                else:
                    output_1d.append(block_2d[i - j - 1][j])
                    Counter += 1
            else:
                if (Counter % 2 == 0):
                    output_1d.append(block_2d[i - j - 1][j])
                else:
                    output_1d.append(block_2d[j][i - j - 1])
                    Counter += 1
        if (Condition):
            Condition = False
        else:
            Condition = True

    k = 2
    Counter = 0
    for i in range(9, 18):
        for j in range(k, 9):
            if (Counter % 2 == 0):
                output_1d.append(block_2d[i - j][j - 1])
            else:
                output_1d.append(block_2d[j - 1][i - j])
        k += 1
        Counter += 1
    return output_1d


def run_length_encoder(block_1d):
    ## Here we are replace any stream of zeros into [0,Number of zeros in the stream]
    Counter = 0
    Condition = True
    output = []
    for i in range(len(block_1d)):
        if (block_1d[i] == 0):
            Counter +=1
            Condition = True
        else:
            if (Counter > 0):
                output.append(0)
                output.append(Counter)
                output.append(block_1d[i])
                Condition = False
                Counter = 0
            else:
                output.append(block_1d[i])
        if(Counter > 0 and i == len(block_1d)-1):
            output.append(0)
            output.append(Counter)

    return np.array(output).astype(int)

def run_length_decoder(Array):
    ## Replacing [0,Number of zeros] -> Number of zeros*[0]
    output = []
    for i in range(len(Array)):
        if (i != 0 and Array[i - 1] == 0):
            for j in range(Array[i] - 1):
                output.append(0)
        else:
            output.append(Array[i])
    return output


def un_zig_zag(array_1d):  # This is applyed for converting 64 array into 8*8block
    output_2d = np.ones((8, 8))
    Counter = 0
    index = 0
    Condition = True
    for i in range(9):
        for j in range(i):
            if (Condition):
                if (Counter % 2 == 0):
                    output_2d[j][i - j - 1] = array_1d[index]
                else:
                    output_2d[i - j - 1][j] = array_1d[index]
                    Counter += 1
            else:
                if (Counter % 2 == 0):
                    output_2d[i - j - 1][j] = array_1d[index]
                else:
                    output_2d[j][i - j - 1] = array_1d[index]
                    Counter += 1
            index += 1
        if (Condition):
            Condition = False
        else:
            Condition = True

    k = 2
    Counter = 0
    for i in range(9, 18):
        for j in range(k, 9):
            if (Counter % 2 == 0):
                output_2d[i - j][j - 1] = array_1d[index]
            else:
                output_2d[j - 1][i - j] = array_1d[index]
            index += 1
        k += 1
        Counter += 1
    return output_2d


def dis_run_length(code):
    output = []
    for i in range(len(code)):
        if (i != 0 and code[i - 1] == 0):
            for j in range(code[i] - 1):
                output.append(0)
        else:
            output.append(code[i])
    return output

def UnZigZag(array_1d):  # This is applyed for converting 64 array into 8*8block
    output_2d = np.ones((8, 8))
    Counter = 0
    index = 0
    Condition = True
    for i in range(9):
        for j in range(i):
            if (Condition):
                if (Counter % 2 == 0):
                    output_2d[j][i - j - 1] = array_1d[index]
                else:
                    output_2d[i - j - 1][j] = array_1d[index]
                    Counter += 1
            else:
                if (Counter % 2 == 0):
                    output_2d[i - j - 1][j] = array_1d[index]
                else:
                    output_2d[j][i - j - 1] = array_1d[index]
                    Counter += 1
            index += 1
        if (Condition):
            Condition = False
        else:
            Condition = True

    k = 2
    Counter = 0
    for i in range(9, 18):
        for j in range(k, 9):
            if (Counter % 2 == 0):
                output_2d[i - j][j - 1] = array_1d[index]
            else:
                output_2d[j - 1][i - j] = array_1d[index]
            index += 1
        k += 1
        Counter += 1
    return output_2d


def dis_reorder(code_1d, block_size, img_shape):
    output_frame = np.zeros(tuple(img_shape))
    index = 0
    for anchor_row in range(0, output_frame.shape[0], 8):
        for anchor_col in range(0, output_frame.shape[1], 8):
            sub_code_1d = code_1d[index:index+block_size[0]*block_size[1]]
            sub_matrix = un_zig_zag(sub_code_1d)
            output_frame[anchor_row: anchor_row+block_size[0], anchor_col: anchor_col+block_size[1]] = sub_matrix
            index += block_size[0]*block_size[1]
    return output_frame