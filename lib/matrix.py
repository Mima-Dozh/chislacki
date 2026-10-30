def print_float(value, *args, **kwargs):
    print(f"{value:.4f}", **kwargs)

def print_vector(answers, name):
    print(f"{name}:", end="\n")
    for i in range(len(answers)):
        if type(answers[i]) == type([]):
            print_float(answers[i][0], end=" ")
            continue
        print_float(answers[i], end=" ")
    print()
    print()

def print_matrix(matrix, name):
    print(f"{name} matrix:", end="\n")
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print_float(matrix[i][j], end=" ")
        print()
    print()

def get_det_diagonal(matrix):
    det = 1
    for i in range(len(matrix)):
        det *= matrix[i][i]
    return det

def get_eye(size):
    return [[1 if i == j else 0 for i in range(size)] for j in range(size)]

def cross_product(x, y):
    matrix = [[0 for i in range(len(y[0]))] for j in range(len(x))]
    for i in range(len(x)):
        for j in range(len(y[0])):
            for k in range(len(x[i])):
                matrix[i][j] += x[i][k] * y[k][j]
    return matrix

def check_inverse_matrix(x, inverse_x):
    size = len(x)
    eye = cross_product(x, inverse_x)
    for i in range(size):
        for j in range(size):
            if i == j and abs(1 - eye[i][j]) > 1e-6 or i != j and abs(eye[i][j]) > 1e-6:
                print(eye)
                raise RuntimeError(f"Inverse matrix is not correct")

def check_answers(x, y, answers):
    for i in range(len(x)):
        result = 0
        for j in range(len(x[i])):
            result += x[i][j] * answers[j][0]
        if abs(result - y[i][0]) > 1e-6:
            raise RuntimeError(f"Answers are not correct {result} != {y[i][0]}")

def get_T(x):
    return [[x[i][j] for i in range(len(x))] for j in range(len(x[0]))]