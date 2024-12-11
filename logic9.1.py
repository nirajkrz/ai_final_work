from itertools import product


# Define logical operations
def NOT(x):
    return 1 - x


def AND(x, y):
    return x & y


def OR(x, y):
    return x | y


def IMPLIES(x, y):
    return OR(NOT(x), y)


def IFF(x, y):
    return (x & y) | (NOT(x) & NOT(y))


# Initialize partial truth table (CS, AW, LQ, HM)
partial_truth_table = [
    (0, 0, 0, 1),  # A
    (0, 1, 0, 0),  # B
    (0, 1, 0, 1),  # C
    (0, 1, 1, 1),  # D
    (1, 0, 0, 0),  # E
    (1, 0, 1, 0),  # F
    (1, 1, 1, 0),  # G
    (1, 1, 1, 1),  # H
]

# Evaluate the circuit for each row in the partial truth table
results = []

for row in partial_truth_table:
    CS, AW, LQ, HM = row

    I1 = OR(CS, AW)
    I2 = NOT(AND(AW, NOT(LQ)))
    I3 = AND(CS, IMPLIES(HM, LQ))
    I4 = AND(AW, IFF(LQ, NOT(CS)))
    I5 = OR(I2, I3)
    I6 = AND(I4, I3)
    Desirable = OR(NOT(I5), NOT(I6))

    results.append((CS, AW, LQ, HM, Desirable))

print(results)
