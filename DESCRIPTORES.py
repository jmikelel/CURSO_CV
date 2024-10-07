import cv2 as cv
import numpy as np
import argparse

def argument():
    parser = argparse.ArgumentParser(description="Procesado de imagenes.")
    parser.add_argument('image_path', type=str, help="Ruta de la imagen")
    return parser.parse_args()

def open_imag(img):
    """
    Muestra la imagen pasada como argumento
    """
    cv.imshow("Imagen", img)
    cv.waitKey(0)

# Metodo SIFT (Scale-Invariant Feature Transform)
def funcion_SIFT(img):
    """
    Aplica el algoritmo SIFT para detectar puntos clave en la imagen.
    Dibuja y guarda dos versiones: una con los puntos clave y otra con más detalles.
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convertir a escala de grises
    sift = cv.SIFT_create()  # Crear el objeto SIFT
    kp = sift.detect(gray, None)  # Detectar los keypoints

    # Dibuja los keypoints en la imagen en escala de grises
    img_keypoints = cv.drawKeypoints(gray, kp, None)
    cv.imwrite('sift_keypoints.jpg', img_keypoints)  # Guardar la imagen con keypoints

    # Dibuja los keypoints con más detalles
    img_rich_keypoints = cv.drawKeypoints(gray, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite('sift_more_keypoints.jpg', img_rich_keypoints)  # Guardar la imagen con más detalles

    cv.imshow("Imagen con Keypoints", img_keypoints)
    cv.waitKey(0)
    
    cv.imshow("Imagen con Keypoints", img_rich_keypoints)
    cv.waitKey(0)
    cv.destroyAllWindows()

def harris_corner(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst=cv.cornerHarris(gray,2,3,0.04)

    dst=cv.dilate(dst,None)

    img[dst>0.01*dst.max()]=[0,0,225]

    cv.imshow('dst',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def SURFing(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    surf = cv.xfeatures2d.SURF_create(400)
    surf.setHessianThreshold(50000)

    kp = surf.detectAndCompute(img,None)
    img_key_point = cv.drawKeypoints(gray,kp,None)
    
    cv.imshow("Imagen con Keypoints", img_key_point)
    cv.waitKey(0)
    cv.destroyAllWindows()

    

def main():
    args = argument()

    image = cv.imread(args.image_path)  # Leer la imagen desde la ruta proporcionada
    if image is None:
        print("Error: No se encontró la imagen en la ruta proporcionada")
        return

    open_imag(image)  # Muestra la imagen original
    harris_corner(image)
    funcion_SIFT(image)  # Aplica SIFT y muestra la imagen modificada
    SURFing(image)
    

if __name__ == "__main__":
    main()
