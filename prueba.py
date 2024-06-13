import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, measure
from scipy.interpolate import splev, splrep

# Cargar la imagen
imagen = io.imread('gourd3.bmp')

# Verificar si la imagen está en RGB, de lo contrario convertirla
if len(imagen.shape) == 3 and imagen.shape[2] == 3:
    imagen_gris = color.rgb2gray(imagen)
else:
    imagen_gris = imagen  # La imagen ya está en escala de grises

# Umbralizar la imagen
umbral = filters.threshold_otsu(imagen_gris)
imagen_binaria = imagen_gris > umbral

# Detectar contornos
contornos = measure.find_contours(imagen_binaria, level=0.8)

# Crear una nueva figura para mostrar la imagen binaria con los contornos
plt.figure()
plt.imshow(imagen_binaria, cmap='gray')
plt.title('Todos los contornos detectados')
for contorno in contornos:
    plt.plot(contorno[:, 1], contorno[:, 0], 'r', linewidth=2)
plt.show()

# Determinar el tamaño de la imagen
alto, ancho = imagen_binaria.shape

# Crear una nueva figura para mostrar los contornos interpolados
plt.figure()
plt.imshow(imagen, cmap='gray')
plt.title('Contornos interpolados')
for contorno in contornos:
    # Calcular el área del contorno
    area_contorno = np.abs(np.trapz(contorno[:, 1], contorno[:, 0]))
    
    # Si el área del contorno es significativamente menor que el área de la imagen
    if area_contorno < alto * ancho * 0.9:  # Ajustar este umbral según sea necesario (90%)
        t = np.arange(contorno.shape[0])
        ts = np.linspace(0, t.max(), 1000)
        
        # Ajustar un spline a los puntos del contorno
        spline_x = splev(ts, splrep(t, contorno[:, 1]))
        spline_y = splev(ts, splrep(t, contorno[:, 0]))
        
        # Dibujar el spline ajustado
        plt.plot(spline_x, spline_y, 'b', linewidth=2)

plt.axis('equal')
plt.axis('off')
plt.show()
