import cv2 as cv
import argparse

def open_imag(img):
    """
    Muestra la imagen pasada como argumento
    """
    cv.imshow("Imagen", img)



def c_2_gray(img):
    return cv.cvtColor(img, cv.COLOR_RGB2GRAY)

def c_2_hsv(img):
    return cv.cvtColor(img, cv.COLOR_RGB2HSV)

def apply_blur(img):
    return cv.GaussianBlur(img,(15,15),0)

def argument():
    parser=argparse.ArgumentParser(description="Procesado de imagenes.")
    parser.add_argument('image_path',
                        type=str,
                        help="Ruta de la imagen")
    parser.add_argument('opcion',
                        type=int,
                        help="Opcion de modificacion")
    return parser.parse_args()


switch_case={
    1: c_2_gray,
    2: c_2_hsv,
    3: apply_blur
}

def main():
    args=argument()
    
    image=cv.imread(args.image_path)
    

    if image is None:
        print("Error: No encontre imagen en la ruta")
        return
    
    open_imag(image)

    modification_function=switch_case.get(args.opcion, lambda img: img)
    #lambda es una forma de crear funciones anonimas o pequenha y rapidas que no requieren ser
    #definidas por 'def'
    # lambda argumento: expresion (lo que recibe-la operacion o valor que devuelve)

    modified_image=modification_function(image)

    cv.imshow("Imagen Modificada", modified_image)
    cv.waitKey(0)
    cv.destroyAllWindows()



if __name__=="__main__":
    main()