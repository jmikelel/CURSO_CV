import cv2 as cv
import numpy as np

## Queremos detectar los colores usando espacios de color HSV
## Rojo 
#[BLUE,GREEN,RED]
#red = np.uint8([[[0,0,255 ]]])
#hsv_red = cv.cvtColor(red,cv.COLOR_BGR2HSV)
#print( hsv_red )

# Límite para rojo [0,100,100]-[10,255,255]
# Límite para rojo [160,100,100]-[180,255,255]
#Depende del usuario cual usar

def seguimiento():
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("No se detectó la cámara")
        return

    while True:
        # Captura el fotograma de la cámara
        _, frame = cap.read()

        # Convierte el fotograma a espacio de color HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Define el rango de color rojo en HSV
        lower_red = np.array([160, 100, 100])
        upper_red = np.array([180, 255, 255])

        # Crea una máscara para el color rojo
        mask = cv.inRange(hsv, lower_red, upper_red)

        # Aplica la máscara para obtener la parte de la imagen con el color rojo
        res = cv.bitwise_and(frame, frame, mask=mask)

        # Muestra las ventanas con la imagen original, la máscara y el resultado
        cv.imshow('frame', frame)
        cv.imshow('mask', mask)
        cv.imshow('res', res)

        # Espera a que se presione la tecla 'ESC' (27) para salir
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break

    # Libera la cámara y cierra las ventanas
    cap.release()
    cv.destroyAllWindows()

def main():
    seguimiento()

if __name__ == "__main__":
    main()
