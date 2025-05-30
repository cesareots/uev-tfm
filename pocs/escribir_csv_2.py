import os
import csv

def escribir_csv(nombre_archivo, index, game):
    """
    Escribe un registro (index, game) en un archivo CSV.

    Si el archivo no existe, lo crea con la cabecera 'index,game'.
    Si el archivo existe, agrega el registro en una nueva línea.

    Args:
        nombre_archivo (str): La ruta y nombre del archivo CSV.
        index (str o int): El valor para la columna 'index'.
        game (str): El valor para la columna 'game'.
    """
    cabecera = ['index', 'game']
    archivo_existe = os.path.exists(nombre_archivo)

    with open(nombre_archivo, mode='a', newline='', encoding='utf-8') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)

        if not archivo_existe:
            escritor_csv.writerow(cabecera)
        
        escritor_csv.writerow([index, game])

# --- Tus listas ---
label = ["game1.json", "game2.json", "game3.json", "game4.json", "game5.json"]
seleccionados = [0, 1, 3] # Indices de los juegos que quieres escribir

# --- Escribir en el CSV solo los juegos seleccionados ---
nombre_csv = "juegos_seleccionados.csv"

# Opcional: Si quieres asegurarte de que el archivo se crea desde cero cada vez que
# ejecutes este script, puedes borrarlo primero. ¡Ten cuidado con esto en producción!
if os.path.exists(nombre_csv):
    os.remove(nombre_csv)
    print(f"Archivo '{nombre_csv}' existente eliminado para un inicio limpio.")

for indice_seleccionado in seleccionados:
    if 0 <= indice_seleccionado < len(label):
        nombre_juego = label[indice_seleccionado]
        escribir_csv(nombre_csv, indice_seleccionado, nombre_juego)
    else:
        print(f"Advertencia: El índice {indice_seleccionado} está fuera del rango de la lista 'label'.")

print(f"\nSe han escrito los juegos seleccionados en '{nombre_csv}'.")

# Opcional: Leer y mostrar el contenido del CSV para verificar
print("\nContenido de 'juegos_seleccionados.csv':")
with open(nombre_csv, mode='r', newline='', encoding='utf-8') as f:
    print(f.read())
