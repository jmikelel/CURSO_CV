import cv2 as cv
import numpy as np
import argparse

def argument():
    parser=argparse.ArgumentParser(description="Procesado de imagenes.")
    parser.add_argument('image_path',
                        type=str,
                        help="Ruta de la imagen")

    return parser.parse_args()

def translation(img):
    image = cv.imread(img)

    rows, cols = image.shape

    M = np.float32([[1, 0, 100],[0, 1, 50]])
    dst = cv.warpAffine(image, M, (cols,rows))

    cv.imshow('imagen trasladada',dst)

def rotation(img):
    image = cv.imread(img)

    rows, cols = image.shape

    M = cv.getRotationMatrix2D((cols-1)/2.0, (rows-1)/2.0,90,1)
    dst =cv.warpAffine(image, M, (cols, rows))

    cv.imshow('imagen rotada', dst)

def mirror(img):
    image = cv.imread(img)

    rows, cols = image.shape

    mirrored = np.zeros_like(image)

    for i in range(rows):
        for j in range(cols):
            mirrored[i, j] = image[i, cols-1-j]
    cv.imshow('imagen espejo', mirrored)




def main():
    args = argument()
    translation(args.image_path)
    rotation(args.image_path)
    mirror(args.image_path)


if __name__ == "__main__":
    main()
