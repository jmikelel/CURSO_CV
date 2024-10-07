import cv2 as cv
import numpy as np
import argparse
import json

def parse_args():
    """Función para analizar argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description=
                                     'Medir objetos utilizando la cámara o un video.')
    parser.add_argument('--distance', type=float, 
                        required=True, 
                        help='Distancia del lente al objeto a medir (en cm).')
    parser.add_argument('--video', type=str, 
                        help='Ruta del video. Si no se especifica, usará la webcam.')
    parser.add_argument('--json', type=str, 
                        required=True, 
                        help='Ruta del archivo JSON con parámetros de corrección.')
    return parser.parse_args()

def load_calibration_data(filename):
    """Carga los datos de calibración desde un archivo JSON."""
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data['objpoints'], data['imgpoints']

def show_video(video_source, mtx, dist, distance):
    """Función para mostrar el video o la webcam."""
    cap = cv.VideoCapture(video_source)
    
    points = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted_frame = cv.undistort(frame, mtx, dist, None, newCameraMatrix)

        cv.imshow('Original', frame)
        cv.imshow('Corregido', undistorted_frame)

        key = cv.waitKey(1)
        if key == ord('q'):
            break

        if cv.getWindowProperty('Corregido', cv.WND_PROP_VISIBLE) >= 1:
            def click_event(event, x, y, flags, param):
                if event == cv.EVENT_LBUTTONDOWN:
                    points.append((x, y))
                    cv.circle(undistorted_frame, (x, y), 5, (0, 255, 0), -1)
                elif event == cv.EVENT_RBUTTONDOWN:
                    points.clear()
                    print("Puntos reiniciados.")

            cv.setMouseCallback('Corregido', click_event)

        for point in points:
            cv.circle(undistorted_frame, point, 5, (0, 255, 0), -1)
        
        if len(points) > 2:
            hull = cv.convexHull(np.array(points))
            area = cv.contourArea(hull)
            perimeter = cv.arcLength(hull, True)

            # Estimar el tamaño en cm basado en la distancia
            # Suponiendo que el tamaño en píxeles es proporcional a la distancia
            # Factor de escala estimado (puedes ajustar este valor según tus pruebas)
            scale_factor = distance / 100  # 1 cm = 1 unidad en este caso

            # Convertir área a unidades reales estimadas
            area_real_estimada = area * (scale_factor ** 2)

            # Mostrar resultados
            cv.putText(undistorted_frame, f'Vertices: {len(points)}', (10, 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv.putText(undistorted_frame, f'Perímetro: {perimeter:.2f}', (10, 60), 
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv.putText(undistorted_frame, f'Area: {area:.2f} píxeles²', (10, 90), 
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv.putText(undistorted_frame, f'Area estimada: {area_real_estimada:.2f} cm²', 
                       (10, 120), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cap.release()
    cv.destroyAllWindows()

def main():
    args = parse_args()
    video_source = 0 if args.video is None else args.video

    # Cargar datos de calibración
    objpoints, imgpoints = load_calibration_data(args.json)

    # Matriz de cámara y coeficientes de distorsión
    mtx = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])  # Matriz ficticia
    dist = np.zeros((5, 1))  # Coeficientes ficticios

    show_video(video_source, mtx, dist, args.distance)

if __name__ == "__main__":
    main()
