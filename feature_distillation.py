import numpy as np
import torch
from scipy.fftpack import dct, idct
from PIL import Image

T = np.array([
    [0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536],
    [0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904],
    [0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619],
    [0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, -0.4157],
    [0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536],
    [0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778],
    [0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913],
    [0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975]
])


num = 8
q_table = np.ones((num, num)) * 30
q_table[0:4, 0:4] = 25


def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def FD_jpeg_encode(input_matrix):
    # 输入是(n,c,h,w)
    output = []
    input_matrix = input_matrix * 255
    n = input_matrix.shape[0]
    # 转为(n,h,w,c)
    input_matrix = input_matrix.transpose(1, 2)
    input_matrix = input_matrix.transpose(2, 3)
    input_matrix = np.array([np.array(Image.fromarray(np.uint8(input_matrix[i])).resize((304, 304))) for i in range(n)])

    h = input_matrix.shape[1]
    w = input_matrix.shape[2]
    c = input_matrix.shape[3]
    horizontal_blocks_num = w / num
    output2 = np.zeros((c, h, w))
    output3 = np.zeros((n, 3, h, w))
    vertical_blocks_num = h / num
    n_block = np.split(input_matrix, n, axis=0)
    for i in range(0, n):
        c_block = np.split(n_block[i], c, axis=3)
        j = 0
        for ch_block in c_block:
            vertical_blocks = np.split(ch_block, vertical_blocks_num, axis=1)
            k = 0
            for block_ver in vertical_blocks:
                hor_blocks = np.split(block_ver, horizontal_blocks_num, axis=2)
                m = 0
                for block in hor_blocks:
                    block = np.reshape(block, (num, num))
                    block = dct2(block)
                    # quantization
                    table_quantized = np.matrix.round(np.divide(block, q_table))
                    table_quantized = np.squeeze(np.asarray(table_quantized))
                    # de-quantization
                    table_unquantized = table_quantized * q_table
                    IDCT_table = idct2(table_unquantized)
                    if m == 0:
                        output = IDCT_table
                    else:
                        output = np.concatenate((output, IDCT_table), axis=1)
                    m = m + 1
                if k == 0:
                    output1 = output
                else:
                    output1 = np.concatenate((output1, output), axis=0)
                k = k + 1
            output2[j] = output1
            j = j + 1
        output3[i] = output2
    # 转为(n,h,w,c)
    output3 = output3.transpose(0, 2, 3, 1)
    output3 = np.array([np.array(Image.fromarray(np.uint8(output3[i])).resize((299, 299))) for i in range(n)])
    output3 = output3 / 255
    output3 = np.clip(np.float32(output3), 0.0, 1.0)
    # 转为(n,c,h,w)
    output3 = output3.transpose(0, 3, 1, 2)
    output3 = torch.from_numpy(output3)
    return output3
