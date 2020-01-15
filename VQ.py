import numpy as np
import cv2
import math, struct
from sklearn.cluster import KMeans
import time, os

filename = 'BaboonRGB.BMP'
clusters = 256

def makeBase(N):
    A = np.array([[0.0 for i in range(8)]for j in range(N)])
    for i in range(len(A)):
        for j in range(len(A)):
            if i == 0:
                A[i][j] = math.sqrt(1 / N) * math.cos(((2 * j + 1) * i * math.pi) / (2 * N))
            else:
                A[i][j] = math.sqrt(2 / N) * math.cos(((2 * j + 1) * i * math.pi) / (2 * N))
    return A
    
def blocking(matrix, isgrayscale):
    block = []
    if (matrix[0:len(matrix), 0:len(matrix), 0] == matrix[0:len(matrix), 0:len(matrix), 1]).all():
        isgrayscale = True
    for color in range(3):
        for i in range(0, len(matrix), 8):
            for j in range(0, len(matrix[0]), 8):
                block.append(matrix[i:i+8, j:j+8, color])   #先把 R 先列再行分成 8X8 大小的區塊，再來 G，再來 B
        if isgrayscale:
            break
    return block, isgrayscale

def unblocking(block, m, n, isgrayscale = False):
    img = np.zeros((m, n, 3))
    k = 0
    for color in range(3):
        for i in range(0, m, 8):
            for j in range(0, n, 8):
                img[i:i+8, j:j+8, color] = block[k]
                k += 1
        if isgrayscale:
            k = 0
    return img
    
def DCT(block):
    A = makeBase(8)
    transformed = np.zeros((0,64))
    for i in range(len(block)):
        temp = np.around(np.dot(A, np.dot(block[i], np.linalg.inv(A))), decimals = 0, out = None)
        transformed = np.concatenate((transformed, zig_zag(temp)))
    return transformed

def IDCT(trans_code):
    A = makeBase(8)
    block = []
    for i in range(len(trans_code)):
        y = zig_zag(trans_code[i], True)
        block.append(np.around(np.dot(np.linalg.inv(A), np.dot(y, A)), decimals = 0, out=None))
    return block
    
def zig_zag(matrix, inverse = False):
    seq = []
    if inverse:
        seq = [[0 for i in range(8)]for j in range(8)]
    i = 0
    j = 0
    k = 0
    flag = 1 #向右上
    while i < 8 and j < 8:
        if inverse:
            seq[i][j] = matrix[k]
            k += 1
        else:
            seq.append(matrix[i][j])
        if i == len(matrix) - 1 and flag == 0:
            j += 1
            flag = 1
        elif j == len(matrix) - 1 and flag == 1:
            i += 1
            flag = 0
        elif j == 0 and flag == 0:
            i += 1
            flag = 1
        elif i == 0 and flag == 1:
            j += 1
            flag = 0
        elif flag == 1:
            i -= 1
            j += 1
        else:
            i += 1
            j -= 1
    if inverse:
        seq = np.asarray(seq)
    else:
        seq = np.asarray(seq).reshape(1,64)
    return seq
    
def Quantizer(X):
    clf = KMeans(n_clusters = clusters, algorithm = 'elkan')
    code = clf.fit_predict(X)
    header = clf.cluster_centers_
    f = open(filename.split(".")[0] + '_encode.' + filename.split(".")[1], 'wb')    #輸出壓縮檔，目前為不包含標頭的版本
    for i in range(len(code)):
        f.write(struct.pack('>B',code[i]))
    f.close()
    return header
    
def Dequantizer(header):
    code = []
    compressed_file = open(filename.split(".")[0] + '_encode.' + filename.split(".")[1], 'rb')
    while True:    #將壓縮檔內容程式碼轉成程式碼列表
        element = compressed_file.read(1)
        if element == b'':    #讀取檔案結束判斷
            break
        (element, ) = struct.unpack('>B',element)
        code.append(element)
    compressed_file.close()
    code = np.asarray(code)
    X = np.zeros((0, 64))
    for index in code:
        X = np.concatenate((X, header[index].reshape(1,64)))
    return X

if __name__ == '__main__':
    global isgrayscale
    isgrayscale = False
    tStart = time.time()
    img = cv2.imread(filename)
    (m, n, z) = img.shape
    print(img.shape)
    block, isgrayscale = blocking(img, isgrayscale)
    transformed = DCT(block)
    #print(len(block))
    print('DCT Complete!')
    header = Quantizer(transformed)
    print('Quantizer Complete!')
    trans_code = Dequantizer(header)
    print('Dequantizer Complete!')
    block = IDCT(trans_code)
    print('IDCT Complete!')
    img2 = unblocking(block, m, n, isgrayscale)
    cv2.imwrite(filename.split(".")[0] + '_decode.BMP', img2)
    tEnd = time.time()
    print('Time:{}'.format(tEnd - tStart))
    
    source_file_size = os.path.getsize(filename)
    compressed_file_size = os.path.getsize(filename.split(".")[0] + '_encode.' + filename.split(".")[1])
    print('壓縮前大小:{}'.format(source_file_size))
    print('壓縮後大小:{}'.format(compressed_file_size))
    print('壓縮率:{}'.format(source_file_size/compressed_file_size))
    
    dif = img2 - img
    mse = np.sum(dif *dif) / (m * n)
    print('失真率:{}'.format(mse))
        
    
    