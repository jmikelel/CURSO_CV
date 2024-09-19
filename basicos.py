import cv2 as cv
import numpy as np
import argparse

def openg_img(imagen1, imagen2):
    img1 = cv.imread(imagen1)
    img2 = cv.imread(imagen2)

    # Mostrar las imágenes originales
    cv.imshow("Imagen 1", img1)
    cv.imshow("Imagen 2", img2)

def resize_images(imagen1, imagen2):
    # Leer las imágenes
    img1 = cv.imread(imagen1)
    img2 = cv.imread(imagen2)

    # Ajustar el tamaño de la segunda imagen al tamaño de la primera
    img2_resized = cv.resize(img2, (img1.shape[1], img1.shape[0]))
    
    return img1, img2_resized

def adition(imagen1, imagen2):
    # Redimensionar las imágenes para que tengan el mismo tamaño
    img1, img2 = resize_images(imagen1, imagen2)

    # Realizar la adición ponderada
    dst = cv.addWeighted(img1, 0.7, img2, 0.3, 0)
    
    # Mostrar el resultado
    cv.imshow('Resultado', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

def argument():
    parser = argparse.ArgumentParser(description="Procesar dos imágenes.")
    parser.add_argument('image_path_1', 
                        type=str, 
                        help="Ruta de la primera imagen")
    parser.add_argument('image_path_2', 
                        type=str, 
                        help="Ruta de la segunda imagen")
    return parser.parse_args()

def main():
    args = argument()

    im1 = args.image_path_1
    im2 = args.image_path_2
    openg_img(im1, im2)
    adition(im1, im2)

if __name__ == "__main__":
    main()
