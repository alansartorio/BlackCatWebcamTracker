import numpy as np
import cv2
import time

class VideoCamera:
    imgWidth, imgHeight = resolution = (640, 480)
    damped = np.array((0, 0, imgWidth, imgHeight), dtype = np.float)
    target = damped[:]
    aspectRatio = imgWidth / imgHeight
    padding = 50

    def __init__(self):
        self.video = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

    def __del__(self):
        print("Releasing Camera")
        self.video.release()
        
    def expandAndFixRectangle(self, x, y, w, h):
        x += w / 2
        y += h / 2
        h += self.padding * 2
        w += self.padding * 2
        h = min(self.imgHeight, h)
        w = min(self.imgWidth, w)
        if h * self.aspectRatio > w:
            w = h * self.aspectRatio
        elif h * self.aspectRatio < w:
            h = w / self.aspectRatio
        
        x -= w / 2
        y -= h / 2

        x, y, w, h = map(int, (x, y, w, h))
        
        if y < 0:y = 0
        if y + h > self.imgHeight: y = self.imgHeight - h
        if x < 0:x = 0
        if x + w > self.imgWidth: x = self.imgWidth - w
        
        return x, y, w, h

    def get_frame(self):

        success, image = self.video.read()                 

        if success:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float) / 255
            
            color = ((hsv[:,:,1] * hsv[:,:,2]) < 0.2)
            dark = (hsv[:,:,2] < 0.2)
            cat = (color * dark).astype(np.uint8)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
            cat = cv2.morphologyEx(cat, cv2.MORPH_OPEN, kernel)
            cat *= 255

            cat = cv2.blur(cat, (5, 5))
            
            contours, hier = cv2.findContours(cat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                cat = max(contours, key = cv2.contourArea)
                cv2.drawContours(image, contours, -1, (0, 0, 255))
                x, y, w, h = self.expandAndFixRectangle(*cv2.boundingRect(cat))
                w, h = map(lambda x:max(10, x), (w, h))
            else:
                x, y, w, h = 0, 0, self.imgWidth, self.imgHeight

            pos = np.array((x, y, w, h), dtype=np.float)
            dist = pos - self.target
            d = np.average(dist**2)
            if d > 20:
                self.target = pos

            self.damped += (self.target - self.damped) / 2
            x, y, w, h = map(int, self.damped)
            w = int(h * self.aspectRatio)

            cat_image = image[y:y+h, x:x+w, :]
            cat_image = cv2.resize(cat_image, self.resolution)
            image = cat_image

        if not success:
            image = np.zeros((480, 640), dtype=np.float)
            print("Errorcito")
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()