import pennylane as qml
import torch
import torch.nn as nn

dev = qml.device("default.qubit", wires=4)

# diff_method ko "backprop" kar diya hai taaki batch training smoothly chale
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(4))
    qml.StronglyEntanglingLayers(weights, wires=range(4))
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

class QMLHybridClassifier(nn.Module):
    def __init__(self, n_features, n_qubits=4, n_classes=2):
        super(QMLHybridClassifier, self).__init__()
        
        self.n_qubits = n_qubits
        self.fc1 = nn.Linear(n_features, n_qubits)
        
        weight_shapes = {"weights": (2, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        self.fc2 = nn.Linear(n_qubits, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(x) * 1.5708
        x = self.q_layer(x)
        x = self.fc2(x)
        return self.softmax(x)