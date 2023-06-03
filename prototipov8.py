# Imporart libreria NumPy para utilizar funciones y estructuras de datos numéricas eficientes.
import numpy as np

# Representar una red de Hopfield
class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons #cantidad de neuronas
        self.weights = np.zeros((num_neurons, num_neurons)) #matriz de pesos
        self.threshold = 0 #umbral de activación

# Entrenamiento de la red
    def train(self, patterns):
        for pattern_set in patterns:
            for pattern in pattern_set:
                pattern = np.array(pattern).reshape(-1, 1)
                self.weights += np.dot(pattern, pattern.T)

        np.fill_diagonal(self.weights, 0) #establece los elementos de la diagonal de la matriz

# Establecer los estados de una neurona segun su peso    
    def update_neuron(self, neuron, pattern):
        activation = np.dot(self.weights[neuron], pattern) - self.threshold
        if activation >= 0:
            return 1
        else:
            return 0

# Toma la entrada y lo presenta a la red de Hopfield, 
# maximo 100 veces puede repetirse
    def recognize(self, pattern, max_iterations=100):
        pattern = np.array(pattern).reshape(-1, 1)
        for _ in range(max_iterations):
            updated_pattern = np.copy(pattern)
            for i in range(self.num_neurons):
                updated_pattern[i] = self.update_neuron(i, updated_pattern)
            if np.array_equal(updated_pattern, pattern):
                return updated_pattern.flatten()
            pattern = updated_pattern

        return None

# toma una imagen representada como una matriz de píxeles 
# y la codifica como una lista de valores binarios
def encode_image(image):
    encoded_image = []
    for row in image:
        encoded_row = [1 if pixel == 1 else -1 for pixel in row]
        encoded_image.extend(encoded_row)
    return encoded_image

#Proceso inverso de encode
def decode_image(encoded_image, width):
    decoded_image = []
    for i in range(0, len(encoded_image), width):
        row = encoded_image[i:i+width]
        decoded_image.append(row)
    return decoded_image

#realizar desplazamiento 
def shift_image(image):
    shifted_image = np.roll(image, shift=1, axis=0)
    shifted_image[0] = [0] * len(image[0])
    return shifted_image

#comparar imagenes
def detect_shift(image, decoded_image):
    if np.array_equal(image, decoded_image):
        print("No hay desplazamiento.")
    else:
        shift_type = None

        # Verificar desplazamiento horizontal
        if any(not np.array_equal(image[i], decoded_image[i]) for i in range(len(image))):
            shift_type = "Desplazamiento horizontal"

        if shift_type is None:
            print("No se detectó ningún desplazamiento.")
        else:
            print(f"La imagen tiene un desplazamiento: {shift_type}.")

        print("\nImagen de entrada:")
        for row in image:
            print(row)

        print("\nImagen reconstruida:")
        for row in decoded_image:
            print(row)


# Definir el bloque del motor (representación de 10x10)
motor_block = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# Definir el bloque de otro patrón (representación de 10x10)
otro_block = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# Codificar los bloques de patrones
encoded_motor_block = encode_image(motor_block)
encoded_otro_block = encode_image(otro_block)

# Crear y entrenar la red de Hopfield con los bloques de patrones
network = HopfieldNetwork(len(encoded_motor_block))
network.train([[encoded_motor_block, encoded_otro_block]])

# Definir la imagen de entrada del bloque del motor (puede ser modificada)
input_image = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# Aplicar desplazamiento en la imagen de entrada
shifted_input_image = shift_image(input_image)

# Codificar la imagen de entrada desplazada del bloque del motor
encoded_input_image = encode_image(shifted_input_image)

# Reconocer la imagen de entrada con la red de Hopfield
reconstructed_image = network.recognize(encoded_input_image)

# Decodificar la imagen reconstruida
decoded_image = decode_image(reconstructed_image, len(shifted_input_image[0]))

# Detectar y mostrar el tipo de desplazamiento y la imagen corregida
detect_shift(shifted_input_image, decoded_image)
