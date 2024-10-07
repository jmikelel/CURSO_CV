import numpy as np
import cv2 as cv
import glob
import argparse
import json
import threading
import sys

# Variable de control global para detener el programa
stop_program = False

def parse_args():
    """Función para analizar argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description=
                                     'Detectar esquinas de un tablero de ajedrez en imágenes.')
    parser.add_argument('image_dir', 
                        type=str, 
                        help='Directorio donde se encuentran las imágenes.')
    parser.add_argument('board_size', 
                        type=int, 
                        nargs=2, 
                        help='Tamaño del tablero (ancho alto).')
    return parser.parse_args()

def prepare_object_points(board_size):
    """Función para preparar los puntos del objeto del tablero de ajedrez."""
    print("Preparando puntos del objeto...")
    objp = np.zeros((board_size[0] * board_size[1], 3), 
                    np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 
                           0:board_size[1]].T.reshape(-1, 2)
    return objp

def find_corners(image_dir, board_size):
    """Función para encontrar esquinas del tablero en las imágenes y mostrarlas."""
    print("Buscando esquinas en las imágenes...")
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = prepare_object_points(board_size)

    objpoints = []
    imgpoints = []
    images = glob.glob(f'{image_dir}/*.jpg')

    for idx, fname in enumerate(images):
        global stop_program
        if stop_program:
            print("Programa detenido manualmente.")
            break

        print(f"Cargando imagen {idx + 1}/{len(images)}: {fname}")
        img = cv.imread(fname)
        if img is None:
            print(f"Error cargando la imagen {fname}. Ignorando.")
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, board_size, None)

        if ret:
            print(f"Esquinas encontradas en imagen index {idx}: {fname}")
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Dibujar y mostrar esquinas en la imagen
            cv.drawChessboardCorners(img, board_size, corners2, ret)
            cv.imshow('img', img)
            if cv.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            print(f"No se encontraron esquinas en imagen index {idx}: {fname}. Ignorando.")
        cv.destroyAllWindows()

    if not objpoints or not imgpoints:
        print("No se encontraron suficientes puntos en las imágenes.")
    return objpoints, imgpoints

def calibrate_camera(objpoints, imgpoints, image_size):
    """Función para calibrar la cámara usando los puntos del objeto y las esquinas detectadas."""
    print("Calibrando cámara...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, image_size, None, None)
    
    print(f"Calibración exitosa: {ret}")
    print(f"Matriz de la cámara:\n{camera_matrix}")
    print(f"Coeficientes de distorsión:\n{dist_coeffs}")

    return camera_matrix, dist_coeffs, rvecs, tvecs

def save_calibration_data(camera_matrix, dist_coeffs, rvecs, tvecs, filename='calibration_data.json'):
    """Función para guardar los parámetros de calibración en un archivo JSON."""
    print(f"Guardando datos de calibración en {filename}...")
    data = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist(),
        'rvecs': [rvec.tolist() for rvec in rvecs],
        'tvecs': [tvec.tolist() for tvec in tvecs]
    }

    with open(filename, 'w') as json_file:
        json.dump(data, json_file)
    print(f'Datos de calibración guardados en {filename}')

def monitor_input():
    """Función que monitorea la entrada de la palabra clave para detener el programa."""
    global stop_program
    while not stop_program:
        user_input = input()
        if user_input.strip().lower() == "mikoto":
            stop_program = True
            print("Deteniendo el programa...")

def main():
    # Iniciar el monitoreo de entrada en un hilo separado
    threading.Thread(target=monitor_input, daemon=True).start()

    args = parse_args()
    print("Iniciando proceso de detección de esquinas y calibración...")
    objpoints, imgpoints = find_corners(args.image_dir, tuple(args.board_size))
    
    if not objpoints or not imgpoints:
        print("No se encontraron suficientes puntos para la calibración.")
        return

    # Obtener el tamaño de la imagen para la calibración
    img = cv.imread(glob.glob(f'{args.image_dir}/*.jpg')[0])
    image_size = (img.shape[1], img.shape[0])  # (ancho, alto)

    # Calibrar la cámara
    camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(objpoints, imgpoints, image_size)

    # Guardar los parámetros de calibración
    save_calibration_data(camera_matrix, dist_coeffs, rvecs, tvecs)

    print("Proceso completado con éxito.")

if __name__ == "__main__":
    main()
