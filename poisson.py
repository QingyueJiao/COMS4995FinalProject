"""
Poisson Image Editing
"""

import scipy.sparse
import scipy.sparse.linalg
import cv2
import numpy as np
import matplotlib.pyplot as plt


def resize(source, target):
    # inputs are both square images
    size = source.shape
    result = cv2.resize(target, (size[1], size[0]))
    return result


def neighbors(pixel, H, W):
    # up, down, left, right -four pixels around
    neighbor = []
    if pixel[0] + 1 < H:
        neighbor.append((pixel[0] + 1, pixel[1]))
    if pixel[0] - 1 >= 0:
        neighbor.append((pixel[0] - 1, pixel[1]))
    if pixel[1] + 1 < W:
        neighbor.append((pixel[0], pixel[1] + 1))
    if pixel[1] - 1 >= 0:
        neighbor.append((pixel[0], pixel[1] - 1))
    return neighbor


def in_delta_omega(mask, pixel):
    # If the pixel is in the mask and some of its neighbor is not in the mask, then it's in delta omega
    H = mask.shape[0]
    W = mask.shape[1]
    if mask[pixel]:
        for neighbor in neighbors(pixel, H, W):
            if not mask[neighbor]:
                return True
    return False


def laplacian(source, mask, target):
    # Construct the laplacian matrix(A and b in Ax = b)
    H = mask.shape[0]
    W = mask.shape[1]
    area = np.nonzero(mask)
    f = []
    for i in range(len(area[0])):
        f.append((area[0][i], area[1][i]))
    N = len(f)

    # Construct A
    A = scipy.sparse.lil_matrix((N, N))
    A.setdiag(4)
    for i in range(N):
        for neighbor in neighbors(f[i], H, W):
            if neighbor in f:
                A[i, f.index(neighbor)] = -1
    A = A.tocsc()

    # Construct b
    b = np.zeros((N, 3))
    for i in range(N):
        for j in range(3):
            b[i][j] = 4 * source[f[i]][j]
            for neighbor in neighbors(f[i], H, W):
                b[i][j] -= source[neighbor][j]
            if in_delta_omega(mask, f[i]):
                for neighbor in neighbors(f[i], H, W):
                    if neighbor not in f:
                        b[i][j] += target[neighbor][j]

    # Solve
    x = scipy.sparse.linalg.spsolve(A, b)
    result = target.copy().astype(int)
    for i in range(N):
        result[f[i]] = x[i]
    return result


def copy_and_paste(source, mask, target):
    area = np.nonzero(mask)
    f = []
    for i in range(len(area[0])):
        f.append((area[0][i], area[1][i]))
    N = len(f)
    result = target.copy().astype(int)
    for i in range(N):
        result[f[i]] = source[f[i]]
    return result


def main():
    # Read in target background
    target = cv2.imread('classic.jpg')
    for i in range(5,6):
        source = cv2.imread('image/source_' + str(i) + '.jpeg')
        mask = cv2.imread('image/mask_' + str(i) + '.jpeg')
        target = resize(source, target)
        result = laplacian(source, mask[:,:,0], target)
        cv_result = cv2.seamlessClone(source, target, mask, (60,60), cv2.NORMAL_CLONE)
        naive_result = copy_and_paste(source, mask, target)
        cv2.imwrite('result/' + str(i) + '.jpg', result)
        cv2.imwrite('result/cv2_' + str(i) + '.jpg', cv_result)
        cv2.imwrite('result/naive_' + str(i) + '.jpg', naive_result)
        print('Finish image' + str(i))


if __name__ == "__main__":
    main()