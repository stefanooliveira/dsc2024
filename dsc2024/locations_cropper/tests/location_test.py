import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'cropped_map.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

height, width, _ = image.shape

lat_min, lat_max = -56, 35
lon_min, lon_max = -116, -25

def latlon_to_pixel(lat, lon, width, height, lat_min, lat_max, lon_min, lon_max):
    x = int((lon - lon_min) / (lon_max - lon_min) * width)
    y = int((lat_max - lat) / (lat_max - lat_min) * height)
    return x, y

# Coordenadas especificadas
# lat = -25.382508523805097
# lon = -49.23480919989463
lat = -55
lon = -110

x, y = latlon_to_pixel(lat, lon, width, height, lat_min, lat_max, lon_min, lon_max)

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.scatter([x], [y],marker='x', c='black', s=100, linewidths=2)  
plt.title(f'Coordenadas: ({lat}, {lon})')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
