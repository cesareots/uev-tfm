def obtener_numeros(rango):
    """Devuelve una lista con todos los números enteros entre los dos valores"""
    inicio, fin = rango

    return list(range(inicio, fin + 1))

print(obtener_numeros([475, 482]))
print(obtener_numeros([417, 417]))

checkpoint_path="models/CNN3D/20250609-192131/model_CNN3D_best.pth"
model_type=str(checkpoint_path).split("/")[1].upper()
print(model_type)
