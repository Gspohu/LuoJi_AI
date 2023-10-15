import random
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
import pickle
import os
from sympy import isprime
import questionary


METADATA = {
    "input_size": 3,
    "hidden_sizes": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    "output_size": 1,
    "epochs": 1000
}


# Activation functions and their derivatives
def tanh_activation(x):
    return math.tanh(x)

def tanh_derivative(x):
    return 1.0 - x**2

def sigmoid_activation(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    sigmoid_value = sigmoid_activation(x)
    return sigmoid_value * (1 - sigmoid_value)

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation_func, activation_derivative_func):
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []
        self.activation = activation_func
        self.activation_derivative = activation_derivative_func

        # Initialize weights and biases
        for i in range(len(self.layer_sizes) - 1):
            self.weights.append([[random.uniform(-1, 1) for _ in range(self.layer_sizes[i + 1])]
                                  for _ in range(self.layer_sizes[i])])
            self.biases.append([random.uniform(-1, 1) for _ in range(self.layer_sizes[i + 1])])

    def forward(self, inputs):
        self.layer_outputs = [inputs]

        # Forward propagation through layers
        for w, b in zip(self.weights, self.biases):
            layer_output = [self.activation(sum([self.layer_outputs[-1][i] * w[i][j] for i in range(len(w))]) + b[j])
                            for j in range(len(b))]
            self.layer_outputs.append(layer_output)

        return self.layer_outputs[-1]

    def backward(self, target_output, learning_rate=0.1):
        # Calculate the error for the last layer (output layer)
        errors = [target_output[i] - self.layer_outputs[-1][i] for i in range(len(target_output))]

        # Backpropagation through layers
        for layer in range(len(self.layer_sizes) - 2, -1, -1):
            deltas = [errors[j] * self.activation_derivative(self.layer_outputs[layer + 1][j]) for j in range(len(errors))]
            errors = [sum([deltas[j] * self.weights[layer][i][j] for j in range(len(deltas))])
                      for i in range(self.layer_sizes[layer])]

            # Update weights and biases
            for i in range(self.layer_sizes[layer]):
                for j in range(self.layer_sizes[layer + 1]):
                    self.weights[layer][i][j] += learning_rate * deltas[j] * self.layer_outputs[layer][i]
                    self.biases[layer][j] += learning_rate * deltas[j]

def interrogate_nn(operation_name):
    config = operations.get(operation_name)
    if not config:
        print(f"Opération {operation_name} non reconnue.")
        return

    nn = load_nn(operation_name)
    if nn is None:
        print(f"Aucun réseau de neurones trouvé pour {operation_name}.")
        return
    else:
        print(f"Le réseau {operation_name} a été chargé.")

    x = int(input("Entrez la valeur de x : "))
    y = int(input("Entrez la valeur de y : "))
    z = int(input("Entrez la valeur de z : "))

    result = nn.forward([x, y, z])[0]
    # Get the expected output
    expected_output = config["function"](x, y, z)

    print(f"Résultat pour {operation_name}: {result}")
    print(f"Résultat attendu : {expected_output}")

def save_nn(nn, operation_name):
    """
    Sauvegarde le réseau de neurones et ses métadonnées dans un fichier.
    """
    filename = f"{operation_name}_nn.pkl"
    data = {
        "metadata": METADATA,
        "weights": nn.weights,
        "biases": nn.biases
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_nn(operation_name):
    """
    Charge le réseau de neurones à partir d'un fichier si les métadonnées correspondent.
    Retourne None si le fichier n'existe pas ou si les métadonnées ne correspondent pas.
    """
    config = operations.get(operation_name)
    if not config:
        return None

    filename = f"{operation_name}_nn.pkl"
    if not os.path.exists(filename):
        return None

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    # Vérification des métadonnées
    if data["metadata"] != METADATA:
        return None

    # Determine activation function and its derivative
    if config["activation_type"] == "tanh":
        activation_func = tanh_activation
        activation_derivative_func = tanh_derivative
    elif config["activation_type"] == "sigmoid":
        activation_func = sigmoid_activation
        activation_derivative_func = sigmoid_derivative
    else:
        raise ValueError(f"Unsupported activation type: {config['activation_type']}")

    # Création et initialisation du réseau de neurones
    nn = NeuralNetwork(METADATA["input_size"], METADATA["hidden_sizes"], METADATA["output_size"],
                              activation_func, activation_derivative_func)
    nn.weights = data["weights"]
    nn.biases = data["biases"]
    return nn

def plot_accuracy(selected_operations):
    accuracies = []

    # Pour chaque opération logique
    for operation_name, config in selected_operations.items():
        # Tentative de chargement du réseau de neurones à partir d'un fichier
        nn = load_nn(operation_name)
        if nn is None:
            print(f"Attention : Aucun réseau de neurones n'a été trouvé pour {operation_name}.")
            accuracies.append(0)
            continue

        correct_predictions = 0
        total_predictions = 0

        # Testez toutes les combinaisons possibles d'entrées
        x_range, y_range, z_range = config.get("range", ([0, 1], [0, 1], [0, 1]))

        for x in range(x_range[0], x_range[1] + 1):
            for y in range(y_range[0], y_range[1] + 1):
                for z in range(z_range[0], z_range[1] + 1):
                    expected_output = config["function"](x, y, z)
                    predicted_output = round(nn.forward([x, y, z])[0])  # Arrondir car la sortie est continue

                    if expected_output == predicted_output:
                        correct_predictions += 1

                    total_predictions += 1

        accuracy = correct_predictions / total_predictions
        accuracies.append(accuracy)

    # Affichez le graphique de précision
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(selected_operations)), accuracies, tick_label=list(selected_operations.keys()))
    plt.ylabel("Accuracy")
    plt.title("Accuracy of Neural Network for Logical Operations")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def display_pca(operation_names):
    # PCA
    NB_COMPONENTS = 12
    pca = PCA(n_components=NB_COMPONENTS)

    # Récupérer uniquement les embeddings des opérations sélectionnées
    embeddings_pca = [load_nn(op).weights[0] for op in operation_names if load_nn(op) is not None]
    if not embeddings_pca:
        print("Aucun réseau de neurones valide trouvé pour les opérations spécifiées.")
        return

    embeddings_pca = np.array(embeddings_pca).reshape(len(embeddings_pca), -1)
    embeddings_pca = pca.fit_transform(embeddings_pca)  # Appliquer PCA sur les embeddings

    # Plot
    ny = int(np.floor(np.sqrt(16/9 * NB_COMPONENTS // 2)))
    nx = int(np.ceil((NB_COMPONENTS //2) / ny))

    plt.figure(figsize=(16/9 * 6 * nx, 6 * ny))
    for n in range(NB_COMPONENTS // 2):
        plt.subplot(nx, ny, n + 1)
        plt.scatter(embeddings_pca[:, 2 * n + 0], embeddings_pca[:, 2 * n + 1], marker='o')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.gca().set_aspect('equal')

        # Annotate each point on the scatter plot
        for i, (x, y) in enumerate(embeddings_pca[:, (2 * n + 0):(2 * n + 2)]):
            plt.text(x, y, operation_names[i], fontsize=12, ha='right', va='bottom')

        plt.xlabel(f"PC {2 * n + 0}")
        plt.ylabel(f"PC {2 * n + 1}")

    plt.tight_layout()
    plt.savefig('pca.png', dpi=300)
    plt.show()


operations = {
    "AND": {
        "function": lambda x, y, z=0: int(x and y),
        "range": ([0, 1], [0, 1], [0, 0]),
        "activation_type": "tanh"
    },
    "OR": {
        "function": lambda x, y, z=0: int(x or y),
        "range": ([0, 1], [0, 1], [0, 0]),
        "activation_type": "tanh"
    },
    "NAND": {
        "function": lambda x, y, z=0: int(not (x and y)),
        "range": ([0, 1], [0, 1], [0, 0]),
        "activation_type": "tanh"
    },
    "NOR": {
        "function": lambda x, y, z=0: int(not (x or y)),
        "range": ([0, 1], [0, 1], [0, 0]),
        "activation_type": "tanh"
    },
    "XOR": {
        "function": lambda x, y, z=0: int(x ^ y),
        "range": ([0, 1], [0, 1], [0, 0]),
        "activation_type": "tanh"
    },
    "XNOR": {
        "function": lambda x, y, z=0: int(not (x ^ y)),
        "range": ([0, 1], [0, 1], [0, 0]),
        "activation_type": "tanh"
    },
    "IMPLICATION": {
        "function": lambda x, y, z=0: int((not x) or y),
        "range": ([0, 1], [0, 1], [0, 0]),
        "activation_type": "tanh"
    },
    "NON-IMPLICATION": {
        "function": lambda x, y, z=0: int(x and (not y)),
        "range": ([0, 1], [0, 1], [0, 0]),
        "activation_type": "tanh"
    },
    "MAJORITY": {
        "function": lambda x, y, z: int((x + y + z) >= 2),
        "range": ([0, 1], [0, 1], [0, 1]),
        "activation_type": "tanh"
    },
    "EVEN_PARITY": {
        "function": lambda x, y, z: int((x + y + z) % 2 == 0),
        "range": ([0, 1], [0, 1], [0, 1]),
        "activation_type": "tanh"
    },
    "ODD_PARITY": {
        "function": lambda x, y, z: int((x + y + z) % 2 == 1),
        "range": ([0, 1], [0, 1], [0, 1]),
        "activation_type": "tanh"
    },
    "AND_OR": {
        "function": lambda x, y, z: int((x and y) or z),
        "range": ([0, 1], [0, 1], [0, 1]),
        "activation_type": "tanh"
    },
    "OR_AND": {
        "function": lambda x, y, z: int((x or y) and z),
        "range": ([0, 1], [0, 1], [0, 1]),
        "activation_type": "tanh"
    },
    "TRIPLE_XOR": {
        "function": lambda x, y, z: int(x ^ y ^ z),
        "range": ([0, 1], [0, 1], [0, 1]),
        "activation_type": "tanh"
    },
    "COMPLEX1": {
        "function": lambda x, y, z: int((x and y) or (x ^ z)),
        "range": ([0, 1], [0, 1], [0, 1]),
        "activation_type": "tanh"
    },
    "COMPLEX2": {
        "function": lambda x, y, z: int((x or y) and (not z and x)),
        "range": ([0, 1], [0, 1], [0, 1]),
        "activation_type": "tanh"
    },
    "COMPLEX3": {
        "function": lambda x, y, z: int((x and not y) or (z ^ y)),
        "range": ([0, 1], [0, 1], [0, 1]),
        "activation_type": "tanh"
    },
    "COMPLEX4": {
        "function": lambda x, y, z: int((x ^ y) and (y ^ z) and (x ^ z)),
        "range": ([0, 1], [0, 1], [0, 1]),
        "activation_type": "tanh"
    },
    "COMPLEX5": {
        "function": lambda x, y, z: int((x and y) ^ (y or z) and (not x)),
        "range": ([0, 1], [0, 1], [0, 1]),
        "activation_type": "tanh"
    },
    "COMPLEX15": {
        "function": lambda x, y, z: int(
            ((x and y) ^ (y or z) and (not x)) or
            ((x or z) and (y ^ z) and (not y)) and
            ((y or x) and (z ^ x) or (not z)) or
            ((x and y) or (not z) and (z or y) and (x ^ y))
        ),
        "range": ([0, 1], [0, 1], [0, 1]),
        "activation_type": "tanh"
    },
    "PRIME_CHECK": {
        "function": lambda x, y, z: isprime(x),
        "range": ([2, 500], [0, 0], [0, 0]),
        "activation_type": "sigmoid"
    }
}


def main():
    while True:
        print("\nMenu:")
        print("1. Interroger un réseau de neurones")
        print("2. Tester l'accuracy d'un réseau de neurones")
        print("3. Afficher le PCA d'un réseau de neurones")
        print("4. Quitter")

        choice = input("Choisissez une option: ")

        if choice == "1":
            operation_name = questionary.select(
                "Sélectionnez l'opération que vous souhaitez interroger",
                choices=[op for op in operations.keys()]
            ).ask()
            if operation_name:  # L'utilisateur a fait un choix
                interrogate_nn(operation_name)
            else:
                print("Aucune opération n'a été sélectionnée.")


        elif choice == "2":
            operation_names = questionary.checkbox(
                "Sélectionnez les opérations",
                choices=[op for op in operations.keys()]
            ).ask()

            selected_operations = {op: operations[op] for op in operation_names}
            plot_accuracy(selected_operations)

        elif choice == "3":
            operation_names = questionary.checkbox(
                "Sélectionnez les opérations",
                choices=[op for op in operations.keys()]
            ).ask()

            selected_operations = {op: operations[op] for op in operation_names}
            display_pca(selected_operations)

        elif choice == "4":
            break
        else:
            print("Choix non reconnu.")

if __name__ == "__main__":
    # Extract weights from the neural network as embeddings
    embeddings = []
    # Pour chaque opération logique
    for operation_name, operation_func in operations.items():
        # Tentative de chargement du réseau de neurones à partir d'un fichier
        nn = load_nn(operation_name)
        if nn is None:
            print(f"Entrainement du réseau de neurones {operation_name}")
            # Création d'un nouveau réseau de neurones si aucun n'est chargé
            nn = NeuralNetwork(METADATA["input_size"], METADATA["hidden_sizes"], METADATA["output_size"])
            x_range, y_range, z_range = operations[operation_name].get("range", ([0, 1], [0, 1], [0, 1]))

            for epoch in range(METADATA["epochs"]):
                for x in range(x_range[0], x_range[1] + 1):
                    for y in range(y_range[0], y_range[1] + 1):
                        for z in range(z_range[0], z_range[1] + 1):
                            output = operation_func(x, y, z)
                            nn.forward([x, y, z])
                            nn.backward([output])
            # Sauvegarde du réseau de neurones entraîné
            save_nn(nn, operation_name)
            print(operation_name+" finish")
        embeddings.append(nn.weights[0])
    main()
