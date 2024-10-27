import cv2 as cv
import numpy as np
import json
import time

## Queremos detectar los colores usando espacios de color HSV
## Rojo 
#[BLUE,GREEN,RED]
#red = np.uint8([[[0,0,255 ]]])
#hsv_red = cv.cvtColor(red,cv.COLOR_BGR2HSV)
#print( hsv_red )

# Límite para rojo [0,100,100]-[10,255,255]
# Límite para rojo [160,100,100]-[180,255,255]
#Depende del usuario cual usar

def draw_grid(frame, rows=3, cols=3):
    """
    Dibuja una cuadrícula en el fotograma dado.

    Args:
        frame: La imagen en la que se dibuja la cuadrícula.
        rows: Número de filas de la cuadrícula (por defecto es 3).
        cols: Número de columnas de la cuadrícula (por defecto es 3).
    """
    height, width, _ = frame.shape
    cell_height = height // rows
    cell_width = width // cols

    # Dibuja líneas horizontales
    for i in range(1, rows):
        y = i * cell_height
        cv.line(frame, (0, y), (width, y), (0, 0, 255), thickness=1)  # Rojo

    # Dibuja líneas verticales
    for j in range(1, cols):
        x = j * cell_width
        cv.line(frame, (x, 0), (x, height), (0, 0, 255), thickness=1)  # Rojo

def get_quadrant(cX, cY, rows=3, cols=3):
    """
    Obtiene el cuadrante de la posición del centroide.

    Args:
        cX: Coordenada X del centroide.
        cY: Coordenada Y del centroide.
        rows: Número de filas en la cuadrícula (por defecto es 3).
        cols: Número de columnas en la cuadrícula (por defecto es 3).

    Returns:
        Un tuple que representa el cuadrante (fila, columna).
    """
    quadrant_row = cY // (480 // rows)  # Suponiendo que la altura es 480
    quadrant_col = cX // (640 // cols)  # Suponiendo que el ancho es 640
    return quadrant_row + 1, quadrant_col + 1  # Ajuste para cuadrantes 1-indexados

def seguimiento():
    """
    Captura video desde la cámara, detecta un objeto rojo y registra datos sobre su posición.

    Guarda los datos en un archivo JSON al estar estacionado en la celda (1, 3) durante 5 segundos.
    """
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("No se detectó la cámara")
        return

    data = []
    estacionado = False
    tiempo_estacionado = None

    while True:
        # Captura el fotograma de la cámara
        _, frame = cap.read()
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Define el rango de color rojo en HSV
        lower_red = np.array([160, 100, 100])
        upper_red = np.array([180, 255, 255])

        # Crea una máscara para el color rojo
        mask = cv.inRange(hsv, lower_red, upper_red)

        # Encuentra los contornos en la máscara
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv.contourArea)

            # Dibuja el polígono en el fotograma
            epsilon = 0.02 * cv.arcLength(largest_contour, True)
            approx = cv.approxPolyDP(largest_contour, epsilon, True)
            cv.polylines(frame, [approx], isClosed=True, color=(0, 255, 0), thickness=2)

            M = cv.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                square_size = 20  # Tamaño del cuadrado
                top_left = (cX - square_size // 2, cY - square_size // 2)
                bottom_right = (cX + square_size // 2, cY + square_size // 2)
                cv.rectangle(frame, top_left, bottom_right, (255, 0, 0), thickness=2)

                cuadrante = get_quadrant(cX, cY)

                # Verifica si el centroide está estacionado en la celda (1, 3)
                if cY // (frame.shape[0] // 3) == 0 and cX // (frame.shape[1] // 3) == 2:
                    if not estacionado:
                        estacionado = True
                        tiempo_estacionado = time.time()
                else:
                    estacionado = False
                    tiempo_estacionado = None

                # Almacena los datos en el diccionario
                entry = {
                    "tiempo": time.time(),
                    "posicion": [cX, cY],
                    "cuadrante": [cuadrante[0], cuadrante[1]],
                    "estacionado": estacionado
                }
                data.append(entry)

        # Dibuja la cuadrícula
        draw_grid(frame)

        # Muestra las ventanas con la imagen original y la máscara
        cv.imshow('frame', frame)
        cv.imshow('mask', mask)

        # Comprueba si está estacionado y han pasado 5 segundos
        if estacionado and tiempo_estacionado and (time.time() - tiempo_estacionado >= 5):
            break

        # Espera a que se presione la tecla 'ESC' (27) para salir
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break

    # Guarda los datos en un archivo JSON
    with open('datos.json', 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

    # Libera la cámara y cierra las ventanas
    cap.release()
    cv.destroyAllWindows()

def main():
    """
    Función principal que inicia el seguimiento de objetos.
    """
    seguimiento()

if __name__ == "__main__":
    main()