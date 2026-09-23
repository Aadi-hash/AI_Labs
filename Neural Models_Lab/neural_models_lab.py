"""
Neural Models Lab Exercise
Based on the uploaded laboratory handout:
"Laboratory – Neural Models: Learning, Depth, Activations, and Output Layers"

This file implements:
    Task 3/4: Binary XOR experiment
        - 2 inputs -> 2 hidden units -> 1 output
        - sigmoid, tanh, and ReLU hidden activations
        - BCEWithLogitsLoss
        - loss/prediction/gradient checks
        - zero-initialisation symmetry experiment
        - activation comparison

    Task 5: Three-class extension
        Class 0: (0,0)
        Class 1: (0,1), (1,0)
        Class 2: (1,1)
        - 2 inputs -> 2 hidden units -> 3 logits
        - CrossEntropyLoss
        - softmax probabilities

CPU is sufficient.
"""

import random
import numpy as np
import torch
import torch.nn as nn


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.set_printoptions(precision=6, sci_mode=False)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

X = torch.tensor(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ],
    dtype=torch.float32,
)

# XOR targets
y_xor = torch.tensor(
    [[0.0], [1.0], [1.0], [0.0]],
    dtype=torch.float32,
)

# Three-class targets:
# 0 -> both inactive
# 1 -> sensors disagree
# 2 -> both active
y_3class = torch.tensor([0, 1, 1, 2], dtype=torch.long)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

class XORNet(nn.Module):
    """
    2 inputs -> 2 hidden units -> output_size outputs.

    For the binary experiment:
        output_size = 1
        output is a logit and BCEWithLogitsLoss is used.

    For the three-class experiment:
        output_size = 3
        outputs are logits and CrossEntropyLoss is used.
    """

    def __init__(self, hidden_activation="sigmoid", output_size=1):
        super().__init__()

        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, output_size)

        if hidden_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif hidden_activation == "tanh":
            self.activation = nn.Tanh()
        elif hidden_activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError("Activation must be sigmoid, tanh, or relu.")

    def forward(self, x):
        h = self.activation(self.fc1(x))
        return self.fc2(h)


# ------------------------------------------------------------
# Helper: evaluate binary XOR network
# ------------------------------------------------------------

def evaluate_binary(model):
    model.eval()

    with torch.no_grad():
        logits = model(X)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()

    return probabilities, predictions


# ------------------------------------------------------------
# Task 4A/B/D:
# Train binary XOR network
# ------------------------------------------------------------

def train_binary(
    activation,
    seed=42,
    steps=5000,
    learning_rate=0.1,
    return_early_gradient=True,
):
    """
    Train one random-initialised XOR network.

    Returns:
        model
        initial_loss
        final_loss
        probabilities
        predictions
        early_gradient_norm
        first_layer_gradient
    """

    torch.manual_seed(seed)
    model = XORNet(
        hidden_activation=activation,
        output_size=1,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initial loss before training
    model.train()
    initial_logits = model(X)
    initial_loss = criterion(initial_logits, y_xor).item()

    early_gradient_norm = None

    for step in range(steps):
        optimizer.zero_grad()

        logits = model(X)
        loss = criterion(logits, y_xor)

        # Reverse-mode automatic differentiation / backpropagation
        loss.backward()

        # Record an early first-layer gradient before updating parameters.
        if step == 0 and return_early_gradient:
            early_gradient_norm = model.fc1.weight.grad.detach().norm().item()

        optimizer.step()

    # Final evaluation
    model.eval()

    with torch.no_grad():
        final_logits = model(X)
        final_loss = criterion(final_logits, y_xor).item()
        probabilities = torch.sigmoid(final_logits)
        predictions = (probabilities >= 0.5).float()

    # Expose one gradient tensor after backward().
    # Do one extra backward pass so parameter.grad is available.
    model.train()
    optimizer.zero_grad()

    logits = model(X)
    loss = criterion(logits, y_xor)
    loss.backward()

    first_layer_gradient = model.fc1.weight.grad.detach().clone()

    return (
        model,
        initial_loss,
        final_loss,
        probabilities.detach(),
        predictions.detach(),
        early_gradient_norm,
        first_layer_gradient,
    )


# ------------------------------------------------------------
# Task 4C:
# Symmetry experiment
# ------------------------------------------------------------

def symmetry_experiment(steps=8, learning_rate=0.1):
    """
    Initialise the network symmetrically.

    All parameters are set to zero so the two hidden units begin
    identically. Their rows in fc1 should remain identical because
    they compute the same function and receive the same gradient.
    """

    torch.manual_seed(SEED)

    model = XORNet(
        hidden_activation="sigmoid",
        output_size=1,
    )

    with torch.no_grad():
        model.fc1.weight.zero_()
        model.fc1.bias.zero_()
        model.fc2.weight.zero_()
        model.fc2.bias.zero_()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print("\n" + "=" * 70)
    print("TASK 4C - SYMMETRY EXPERIMENT")
    print("=" * 70)

    for step in range(steps + 1):
        print(f"\nStep {step}")
        print("fc1 weight matrix:")
        print(model.fc1.weight.detach())

        if step < steps:
            optimizer.zero_grad()

            logits = model(X)
            loss = criterion(logits, y_xor)
            loss.backward()

            optimizer.step()

    row_difference = (
        model.fc1.weight[0] - model.fc1.weight[1]
    ).detach().norm().item()

    print("\nFinal Euclidean distance between hidden-layer rows:")
    print(f"{row_difference:.10f}")

    print(
        "\nInterpretation: with identical zero initialisation, the two hidden "
        "units receive identical gradients, so their parameter rows remain identical."
    )


# ------------------------------------------------------------
# Task 4:
# Activation experiment
# ------------------------------------------------------------

def activation_experiment():
    print("\n" + "=" * 70)
    print("TASK 4D - ACTIVATION EXPERIMENT")
    print("=" * 70)

    results = {}

    for activation in ["sigmoid", "tanh", "relu"]:
        (
            model,
            initial_loss,
            final_loss,
            probabilities,
            predictions,
            early_gradient_norm,
            first_layer_gradient,
        ) = train_binary(
            activation=activation,
            seed=SEED,
            steps=5000,
            learning_rate=0.1,
        )

        correct = int(
            (predictions.squeeze(1) == y_xor.squeeze(1)).sum().item()
        )

        results[activation] = {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "correct": correct,
            "early_gradient_norm": early_gradient_norm,
        }

        print(
            f"{activation:>8} | "
            f"initial loss = {initial_loss:.6f} | "
            f"final loss = {final_loss:.6f} | "
            f"correct = {correct}/4 | "
            f"early ||grad_W1||2 = {early_gradient_norm:.6f}"
        )

    print(
        "\nInterpretation: the experiment compares the observed optimisation "
        "behaviour of the three activations on this four-point XOR dataset. "
        "It does not establish that one activation is universally best."
    )

    return results


# ------------------------------------------------------------
# Task 3/4A/B:
# Detailed successful binary run
# ------------------------------------------------------------

def binary_xor_experiment():
    print("\n" + "=" * 70)
    print("BINARY XOR EXPERIMENT")
    print("=" * 70)

    activation = "sigmoid"

    (
        model,
        initial_loss,
        final_loss,
        probabilities,
        predictions,
        early_gradient_norm,
        first_layer_gradient,
    ) = train_binary(
        activation=activation,
        seed=SEED,
        steps=5000,
        learning_rate=0.1,
    )

    print(f"Hidden activation: {activation}")
    print(f"Initial loss:      {initial_loss:.6f}")
    print(f"Final loss:        {final_loss:.6f}")
    print(f"Early gradient norm: {early_gradient_norm:.6f}")

    print("\nFinal probabilities:")
    for x, p in zip(X, probabilities):
        print(f"Input {x.tolist()} -> P(y=1) = {p.item():.6f}")

    print("\nThresholded predictions:")
    for x, pred, target in zip(X, predictions, y_xor):
        print(
            f"Input {x.tolist()} -> "
            f"prediction = {int(pred.item())}, "
            f"target = {int(target.item())}"
        )

    print("\nFirst-layer gradient after backward():")
    print(first_layer_gradient)

    print(
        "\nGradient meaning: each entry of fc1.weight.grad represents the "
        "partial derivative of the scalar loss with respect to the "
        "corresponding first-layer weight."
    )

    print(
        "\nBecause BCEWithLogitsLoss uses a mean reduction by default, "
        "the gradient is the average contribution of the four examples."
    )

    return model


# ------------------------------------------------------------
# Task 5:
# Three-class extension
# ------------------------------------------------------------

def train_three_class(steps=3000, learning_rate=0.1):
    """
    2 inputs -> 2 hidden units -> 3 logits.

    CrossEntropyLoss internally combines log-softmax with negative
    log likelihood. We do not apply softmax before the loss.
    """

    torch.manual_seed(SEED)

    model = XORNet(
        hidden_activation="sigmoid",
        output_size=3,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(steps):
        optimizer.zero_grad()

        logits = model(X)
        loss = criterion(logits, y_3class)

        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        logits = model(X)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)

    return model, loss.item(), logits, probabilities, predictions


def three_class_experiment():
    print("\n" + "=" * 70)
    print("TASK 5 - THREE-CLASS EXTENSION")
    print("=" * 70)

    print(
        "\nClasses:\n"
        "  0 -> both sensors inactive: (0,0)\n"
        "  1 -> sensors disagree:      (0,1), (1,0)\n"
        "  2 -> both sensors active:   (1,1)"
    )

    model, final_loss, logits, probabilities, predictions = train_three_class()

    print("\nArchitecture:")
    print("2 inputs -> 2 hidden units -> 3 logits")

    print("\nFinal loss:")
    print(f"{final_loss:.6f}")

    print("\nFinal logits and class probabilities:")

    for i, (x, logit, prob, pred, target) in enumerate(
        zip(X, logits, probabilities, predictions, y_3class)
    ):
        print(f"\nInput: {x.tolist()}")
        print(f"Logits: {logit.tolist()}")
        print(f"Probabilities: {prob.tolist()}")
        print(f"Predicted class: {pred.item()}")
        print(f"Target class:    {target.item()}")
        print(f"Probability sum: {prob.sum().item():.8f}")

    # Required numerical softmax check for one example
    example_probabilities = probabilities[0]
    print("\nSoftmax sum check for first example:")
    print(
        f"sum(probabilities) = "
        f"{example_probabilities.sum().item():.8f}"
    )

    # Optional numerical stability diagnostic:
    # Adding the same constant to all logits should not change softmax.
    shifted_logits = logits[0] + 100.0

    original = torch.softmax(logits[0], dim=0)
    shifted = torch.softmax(shifted_logits, dim=0)

    print("\nOptional +100-to-all-logits stability check:")
    print("Original:", original.tolist())
    print("Shifted: ", shifted.tolist())
    print(
        "Maximum absolute difference:",
        torch.max(torch.abs(original - shifted)).item(),
    )

    print(
        "\nFinal-layer shape: (3, 2), because there are 3 output logits "
        "and 2 hidden features. Each example therefore produces 3 logits."
    )

    print(
        "\nThe softmax converts the three logits into nonnegative values "
        "whose sum is 1. For cross-entropy, the gradient with respect to "
        "the logits is p - y (where y is the one-hot target vector)."
    )

    return model


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    print("Neural Models Laboratory")
    print("XOR + backpropagation + symmetry + activation + 3-class extension")

    print("\nDataset:")
    print("X =", X)
    print("XOR targets =", y_xor.squeeze().tolist())

    # Binary XOR / gradient checks
    binary_xor_experiment()

    # Symmetry experiment
    symmetry_experiment()

    # Sigmoid vs tanh vs ReLU
    activation_experiment()

    # Three-class extension
    three_class_experiment()

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
