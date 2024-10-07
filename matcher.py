import cv2 as cv
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def image_paths():
    parser = argparse.ArgumentParser(description="Procesado de imagenes.")
    parser.add_argument('image_path1',
                        type=str,
                        help="Ruta de la imagen_1")
    parser.add_argument('image_path2',
                        type=str,
                        help="Ruta de la imagen_2")
    return parser.parse_args()


def read_image(path1, path2): 
    img1 = cv.imread(path1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(path2, cv.IMREAD_GRAYSCALE)
    
    return img1, img2


def descriptor(im1, im2): 
    sift = cv.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    matchesMask = [[0, 0] for _ in range(len(matches))]
    good_matches = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            good_matches.append(m)

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       flags=cv.DrawMatchesFlags_DEFAULT)
    
    img_matches = cv.drawMatchesKnn(im1, kp1, im2, kp2, matches[:5], None, **draw_params)

    # Display the image using OpenCV
    cv.imshow('Matches', img_matches)
    cv.waitKey(0)  # Wait for a key press to close the window
    cv.destroyAllWindows()  # Close all OpenCV windows


def main():
    args = image_paths()
    im1, im2 = read_image(args.image_path1, args.image_path2)
    descriptor(im1, im2)


if __name__ == '__main__':
    main()
