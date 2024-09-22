from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import tensorflow as tf
import numpy as np
import cv2
import webcolors
from coco_classes import COCO_CLASSES  # Importar las clases COCO

app = FastAPI()

# Cargar el modelo preentrenado desde la carpeta "model"
model = tf.saved_model.load("./model/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model")

def closest_color(requested_color):
    min_colors = {}
    # Obtener la lista de nombres de colores CSS3
    color_names = webcolors.names("css3")
    
    # Iterar sobre los nombres y convertir a hexadecimal
    for name in color_names:
        hex_value = webcolors.name_to_hex(name)  # Convertir el nombre a su valor hexadecimal
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_value)  # Convertir hex a RGB
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_color_name(rgb_color):
    try:
        # Intentamos obtener el nombre exacto del color
        return webcolors.rgb_to_name(rgb_color)
    except ValueError:
        # Si no tiene un nombre exacto, obtenemos el color más cercano
        return closest_color(rgb_color)

def detect_object(image):
    # Convertir la imagen a un tensor de entrada
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Realizar la detección de objetos
    detections = model(input_tensor)

    # Extraer las clases detectadas y las puntuaciones
    detection_classes = detections['detection_classes'].numpy()[0]
    detection_scores = detections['detection_scores'].numpy()[0]
    detection_boxes = detections['detection_boxes'].numpy()[0]

    # Filtrar los objetos con puntuaciones superiores a 0.5
    detected_objects = []
    for i in range(len(detection_scores)):
        if detection_scores[i] > 0.5:
            detected_objects.append({
                'class': COCO_CLASSES.get(int(detection_classes[i]), 'unknown'),  # Convertir a string
                'score': float(detection_scores[i]),  # Convertir a flotante
                'box': detection_boxes[i].tolist()  # Convertir la caja a lista para que sea serializable
            })

    return detected_objects

def get_dominant_color(image):
    # Convertir la imagen a HSV y encontrar el color dominante
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    avg_color_per_row = np.average(hsv_image, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)

    # Convertir los valores HSV a RGB para usar en webcolors
    rgb_color = cv2.cvtColor(np.uint8([[avg_color]]), cv2.COLOR_HSV2RGB)[0][0]
    return rgb_color  # Devolver el color en formato RGB

@app.post("/detect/")
async def detect_object_and_color(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    
    # Convertir la imagen a RGB si tiene 4 canales (RGBA)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image_np = np.array(image)

    # Detectar objetos en la imagen
    detections = detect_object(image_np)

    # Obtener el color dominante para cada objeto
    dominant_colors = []
    for obj in detections:
        # Extraer la región correspondiente al objeto (en coordenadas normales)
        ymin, xmin, ymax, xmax = obj['box']
        height, width, _ = image_np.shape
        (left, right, top, bottom) = (int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height))

        # Extraer la porción de la imagen correspondiente al objeto detectado
        object_image = image_np[top:bottom, left:right]

        # Obtener el color dominante de esa región
        if object_image.size > 0:
            dominant_color = get_dominant_color(object_image)
            # Convertir a nombre de color
            dominant_color_name = get_color_name(tuple(dominant_color))
            obj['dominant_color'] = dominant_color_name

    return {
        "objects": detections  # lista de objetos detectados con nombres y colores
    }