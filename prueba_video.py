from ultralytics import YOLO      #Libreria Ultralytics
import cv2                        #OPEN CV

#Cargar Modelo preentrenado
model = YOLO("yolo11n.pt")     


#MODELO DE DETECCIÓN DE OBJETOS EN FRAMES

# Abrir el video con OpenCV
video_path = "oficina.mp4"
cap = cv2.VideoCapture(video_path)

new_width = 460
new_height = 300

# Verificar que el video se haya abierto correctamente
if not cap.isOpened():
    print("Error al abrir el video.")
    exit()

# Procesar cada frame del video
while True:
    ret, frame = cap.read()            # Leer un frame
    if not ret:                        # Si no hay más frames, salir
        break

    results = model.predict(frame, conf = 0.85)            # Aplicar el modelo al frame, para que muestre solo a los de confianza > 0,85

    frame = results[0].plot()          # Dibuja las cajas y etiquetas sobre el frame
    resized_image = cv2.resize(frame, (new_width, new_height))

    cv2.imshow("Deteccion YOLO", resized_image)  # Mostrar el frame con las detecciones

    # Presionar 'q' para salir del video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()