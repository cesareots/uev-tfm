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
    # Define la cabecera
    cabecera = ['index', 'game']
    
    # Comprueba si el archivo existe
    archivo_existe = os.path.exists(nombre_archivo)

    # Abre el archivo en modo append ('a') para añadir. 
    # newline='' es crucial para evitar líneas en blanco extra en CSV.
    with open(nombre_archivo, mode='a', newline='', encoding='utf-8') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)

        # Si el archivo no existe, escribe la cabecera
        if not archivo_existe:
            escritor_csv.writerow(cabecera)
        
        # Escribe el registro actual
        escritor_csv.writerow([index, game])

# --- Ejemplos de uso ---
if __name__ == "__main__":
    # Ejemplo 1: El archivo no existe, se creará con la cabecera
    print("Primer uso: 'mi_log.csv'")
    escribir_csv("mi_log.csv", 1, "Partido A - Barcelona vs Bayern")
    escribir_csv("mi_log.csv", 2, "Partido B - Real Madrid vs Atleti")

    # Ejemplo 2: El archivo ya existe, se agregarán nuevas líneas
    print("\nSegundo uso: 'mi_log.csv' (añadiendo a un archivo existente)")
    escribir_csv("mi_log.csv", 3, "Partido C - PSG vs Man City")

    # Ejemplo 3: Usando un nuevo archivo para mostrar la creación desde cero de nuevo
    print("\nTercer uso: 'otro_log.csv' (creando un nuevo archivo)")
    escribir_csv("otro_log.csv", 10, "Final Copa - Equipo X vs Equipo Y")

    # Puedes verificar los contenidos de los archivos 'mi_log.csv' y 'otro_log.csv'
    # en el mismo directorio donde ejecutes este script.
