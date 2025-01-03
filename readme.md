Basado en Vista cenital.

Procura identificar las líneas de carril entre todos los segmentos detectados en la imagen.

# Resumen de cambios

- carril.py: primera prueba
- carril2.py: parte del código trasladada a los módulos HUI.py y detector.py
- carril3.py: corrige un problema del detector, agrega variables al print para debug
- carril4.py: agrega vista cenital amplia
- carril5.py: 
  - corrige el planteo de Hough
  - corrige el cómputo de distancias,
  - implementa drawSegments con colorMap
  - clase SegmentsAnnotator reune los métodos de anotación: el código de anotación de detector.py quedó incompatible para carril2.py a carril4.py


# Módulos

- HUI.py: interfaz de usuario para determinar la homografía y el tamaño de la vista cenital
- detector.py:
  - Segments: contiene segmentos y propiedades computadas como Hough
  - Bins: 
  - HoughSpace