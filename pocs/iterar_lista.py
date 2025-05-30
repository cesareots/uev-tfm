########################################
lista_indice_lote = [
    [0, 24],
    [25, 49],
    [50, 74],
    [75, 99],
    [100, 100],
]


def imprimir(rango):
    """Imprime todos los números enteros entre los dos valores de la posición dada"""
    inicio, fin = rango
    for numero in range(inicio, fin + 1):
        print(numero)


imprimir(lista_indice_lote[4])


########################################
def obtener_numeros(rango):
    """Devuelve una lista con todos los números enteros entre los dos valores"""
    inicio, fin = rango
    return list(range(inicio, fin + 1))


numeros = obtener_numeros(lista_indice_lote[4])
print(type(numeros))
print(numeros)

########################################
