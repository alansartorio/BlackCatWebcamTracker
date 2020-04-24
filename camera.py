import numpy as np
import cv2
import time

# Dibuja los contornos de las regiones encontradas
debug = False

class VideoCamera:
    imgWidth, imgHeight = resolution = (640, 480) # ACA poner resolucion de la camara
    damped = np.array((0, 0, imgWidth, imgHeight), dtype = np.float)
    target = damped[:]
    aspectRatio = imgWidth / imgHeight
    border = 50 # Borde al rededor del gato detectado
    dampedConstant = 2 # Mientras mas alta, mas despacio se acerca la camara
    lastFrame = np.zeros(resolution[::-1], dtype=np.float)

    def __init__(self):
        self.video = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

    def __del__(self):
        print("Releasing Camera")
        self.video.release()
        
    def expandAndFixRectangle(self, x, y, w, h):
        """Expande el Bounding Box self.border para cada lado, y corrije el (w, h) para que coincida con self.aspectRatio"""
        # Calculo el (x, y) central del bounding box
        x += w / 2
        y += h / 2
        # Amplio el ancho y el alto para agregarle un borde
        h += self.border * 2
        w += self.border * 2
        # Limito para que no se pase del tamaño de la imagen
        h = min(self.imgHeight, h)
        w = min(self.imgWidth, w)
        # Le doy un minimo tamaño al ancho y al alto de 10pix
        w, h = map(lambda x:max(10, x), (w, h))

        # Recalculo el ancho o el alto para que se mantenga el aspect ratio
        if h * self.aspectRatio > w:
            w = h * self.aspectRatio
        elif h * self.aspectRatio < w:
            h = w / self.aspectRatio
        
        # Recalculo el (x, y) de la esquina
        x -= w / 2
        y -= h / 2

        # Redondeo los valores
        x, y, w, h = map(int, (x, y, w, h))
        
        # Acomodo la BB para que quede dentro de la imagen
        if y < 0:y = 0
        if y + h > self.imgHeight: y = self.imgHeight - h
        if x < 0:x = 0
        if x + w > self.imgWidth: x = self.imgWidth - w
        
        # Retorno la BB acomodada
        return x, y, w, h

    def catDetection(self, frame):
        """Retorna el contorno del gato negro encontrado en el frame, o None si no se encontró."""
        # Convierto de BGR a espacio HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float) / 255
        
        # Saturacion * Valor ~= intensidad de color (Gato negro, no tiene color)
        color = ((hsv[:,:,1] * hsv[:,:,2]) < 0.2)
        # Valor == brillo (Gato negro, no tiene brillo)
        dark = (hsv[:,:,2] < 0.2)
        # En donde se cumplen las dos condiciones, hay grices oscuros o negros
        cat = (color * dark).astype(np.uint8)
        
        # Aplico erosion y dilatacion para reducir ruido
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
        cat = cv2.morphologyEx(cat, cv2.MORPH_OPEN, kernel)
        cat *= 255

        # Aplico blur para unir detecciones cercanas
        cat = cv2.blur(cat, (5, 5))
        
        # Busco regiones
        contours, hier = cv2.findContours(cat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Si se encontro alguna region, retorno la de mas area
        if len(contours) > 0:
            return max(contours, key = cv2.contourArea)
        
        # No se encontraron gatos
        return None
    
    def dampPosition(self, cat_bb):
        pos = np.array(cat_bb, dtype=np.float)
        # Calculo diferencia entre bounding box "ideal" y objetivo
        dist = pos - self.target
        d = np.average(dist**2)
        # Si la diferencia entre las componentes de la BB es suficiente, cambio de objetivo
        # Esto esta hecho para evitar vibraciones excesivas
        if d > 20:
            self.target = pos

        # Acerco el BB actual al BB objetivo
        self.damped += (self.target - self.damped) / self.dampedConstant
        x, y, w, h = map(int, self.damped)
        # Corrijo error en el aspectRatio
        w = int(h * self.aspectRatio)

        return x, y, w, h

    def get_frame(self):
        """Obtengo y proceso frame de la camara"""
        # Leo frame de la camara
        success, frame = self.video.read()                 

        if success:
            cat = self.catDetection(frame)
            if cat is not None:
                if debug:
                    cv2.drawContours(frame, [cat], -1, (0, 0, 255))
                # Obtengo y corrijo el Bounding Box de el gato
                x, y, w, h = self.expandAndFixRectangle(*cv2.boundingRect(cat))
            else:
                # Muestro toda la imagen
                x, y, w, h = 0, 0, self.imgWidth, self.imgHeight

            # Suavizo el recorrido del BB
            x, y, w, h = self.dampPosition((x, y, w, h))

            # Recorto y escalo la imagen para que vuelva al tamaño original
            cat_image = frame[y:y+h, x:x+w, :]
            cat_image = cv2.resize(cat_image, self.resolution)
            frame = cat_image


        # Si no se consiguió frame, se retorna el anterior
        if not success:
            return self.lastFrame
        # Codifico el frame en jpg para ser enviado
        ret, jpeg = cv2.imencode('.jpg', frame)
        currentBytes = self.lastFrame = jpeg.tobytes()
        return currentBytes