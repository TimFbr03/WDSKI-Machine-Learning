import numpy as np
np.random.seed(42)

from neuronal_network import NeuronalNetwork

def make_spiral(n_samples=100):
    t = 0.75 * np.pi * (1 + 3 * np.random.rand(1, n_samples))
    x1 = t * np.cos(t)
    x2 = t * np.sin(t)
    y = np.zeros_like(t)

    t2 = 0.75 * np.pi * (1 + 3 * np.random.rand(1, n_samples))
    x1 = np.hstack([-x1, t2 * np.cos(t2)])
    x2 = np.hstack([-x2, t2 * np.sin(t2)])
    y = np.hstack([y, np.ones_like(t2)])

    X = np.concatenate((x1, x2))
    X += 0.5 * np.random.randn(2, 2 * n_samples)

    return X.T, y[0]

# Netzwerk initialisiern
input_dimension = 2
hidden_dimension = 16
output_dimension = 1
learnin_rate = 0.01

if __name__ == "__main__":
    # Daten generieren
    X, y = make_spiral(100)

    # Netzwerk initialisieren (Eingabedimension = 2)
    nn = NeuronalNetwork(
        input_dimension,
        hidden_dimension,
        output_dimension,
        learnin_rate
    )

    # Training
    nn.train(X, y, epochs=75000, verbose=True)

    # Auswertung
    print("Accuracy:", nn.accuracy(X, y))
    nn.info()