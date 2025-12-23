# Detector de carriles con FLD + espacios de Hough

Este proyecto es una prueba de concepto del uso de espacios de Hough para detectar carriles y otros elementos de interés en la pista de la competencia BFMC.

El proyecto consume un video grabado desde el auto, y produce una vista cenital y un espacio de Hough.  No incluye el análisis del espacio de Hough para reconocer las líneas de carriles.

![Vista cenital](docs/_static/Zenithal_wide.png)

[Video demo](https://drive.google.com/file/d/1K8u8Ec_z8S3TaQzb88h5Vym0D-rSOVIx/view)

[Documentación del proyecto](https://laboratorio-de-robotica.github.io/Carriles-FLD-Hough-BFMC/doc.html)

# Instalación

1. Requisitos: Python3 y Numpy
2. Clonar el repositorio
3. Descargar algún video grabado desde el auto y guardarlo en la carpeta recién clonada
4. Ejecutar `python3 carril.py -v nombre_del_video`

Para usar el video por defecto *bfmc2020_online_2.avi*, ejecutar simplemente: `python3 carril.py`


# Videos desde el auto

La siguiente carpeta tiene varios videos proporcionados por Bosch, grabados en el evento BFMC:

[carpeta de videos](https://drive.google.com/drive/folders/19c24oCuQsnvKA2HSh8SWaYJuEkeUOhlY)

Para probar conviene descargar directamente el video por defecto:
[bfmc2020_online_2.avi](https://drive.google.com/file/d/10GSbrm0QxDPsdfeswEVYxFwTpjKVx5Vv/view?usp=drive_link)


# Módulos

- HUI.py: interfaz de usuario para determinar la homografía y el tamaño de la vista cenital
- detector.py: contiene las siguientes clases
  - Segments: contiene segmentos y propiedades computadas como Hough
  - SegmentsAnnotator
  - Bins
  - HoughSpace

`main.py` y `carril.py` es un ejemplo de uso de estas bibliotecas.

El documento [Carriles Hough](https://docs.google.com/document/d/1SYC7nZT3Y_Wc-rTGiAnix9lpgIxPOmTWTjAN7liq6cs/edit?usp=drive_link) brinda el fundamento teórico de este proyecto.

# Versiones anteriores

Los archivos numerados corresponden a versiones anteriores pero que funcionan, y están destinados a ser eliminados.  Por ejemplo, ya se eliminaron carril1.py a carril4.py.

- carril7.py es la versión más avanzada, usa detector3.py; versiones posteriores se implementan en main.py
- carril5.py y carril6.py: dos implementaciones con cambios profundos, carril.py se basa en carril6.py.  carril7.py está en desarrollo.
- detector1.py: versión anterior de detector.py, usada por carril5.py