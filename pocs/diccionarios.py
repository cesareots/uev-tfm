SOCCERNET_LABELS = {
    "Goal": 0,
    "Yellow card": 1,
    "Substitution": 2,
    "Corner": 3,
}
print(SOCCERNET_LABELS)

INV_LABEL_MAP = {v: k for k, v in SOCCERNET_LABELS.items()}
print(INV_LABEL_MAP)

ACCIONES_RECORTAR = [clave for clave, _ in sorted(SOCCERNET_LABELS.items(), key=lambda item: item[1])]
print(ACCIONES_RECORTAR)
print(ACCIONES_RECORTAR[0])
