
def calculatingParametersLine(x1: int, y1: int, x2: int, y2: int):
    A = y1 - y2
    B = x2 - x1
    C = (x1 * y2) - (x2 * y1)
    return A, B, C

print(calculatingParametersLine(1442, 516, 1502, 515))