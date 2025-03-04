import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets, transforms
import numpy as np

# Hybrid Spiking-Quantized Neuron
class HybridNeuron(nn.Module):
    def __init__(self, threshold=0.3, tau=0.95, init_bits=4, max_bits=8):
        super().__init__()
        self.threshold = threshold
        self.tau = tau
        self.init_bits = init_bits
        self.max_bits = max_bits
        
        # Learnable parameters
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.zero_point = nn.Parameter(torch.tensor(0.0))
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        # State variables
        self.mem = None
        self.spk = None

    def reset(self):
        self.mem = None
        self.spk = None

    def quantize(self, x):
        # Straight-Through Estimator for quantization
        with torch.no_grad():
            spike_density = torch.sigmoid(5 * self.spk.mean()) if self.spk is not None else 0.0
            self.bit_width = int(torch.clamp(
                spike_density * self.max_bits + self.init_bits,
                min=self.init_bits, max=self.max_bits
            ))
            q_min = -2 ** (self.bit_width - 1)
            q_max = 2 ** (self.bit_width - 1) - 1
            x_quant = torch.clamp(torch.round(x / self.scale + self.zero_point), q_min, q_max)
        return (x_quant - self.zero_point) * self.scale + (x - x.detach())

    def forward(self, x):
        if self.mem is None or self.mem.size() != x.size():
            self.mem = torch.zeros_like(x)
            self.spk = torch.zeros_like(x)
            
        # Membrane potential update
        self.mem = self.tau * self.mem + self.quantize(x)
        
        # Surrogate gradient
        surrogate = torch.sigmoid(self.alpha * (self.mem - self.threshold))
        spk = (self.mem >= self.threshold).float()
        self.spk = spk - surrogate.detach() + surrogate
        
        # Reset mechanism
        self.mem = (self.mem - spk * self.threshold).detach()
        return self.spk

# Hybrid SNN-QNN Model
class HybridSNNQNN(nn.Module):
    def __init__(self, num_inputs=784, num_hidden=256, num_outputs=10, time_steps=20):
        super().__init__()
        self.time_steps = time_steps
        
        # Input processing
        self.encoder = nn.Linear(num_inputs, num_hidden)
        self.bn = nn.BatchNorm1d(num_hidden)
        self.dropout = nn.Dropout(0.3)
        
        # Temporal processing
        self.lstm = nn.LSTM(num_hidden, num_hidden//2, batch_first=True)
        self.neuron1 = HybridNeuron(threshold=0.4)
        self.neuron2 = HybridNeuron(threshold=0.3)
        
        # Classification
        self.decoder = nn.Linear(num_hidden//2, num_outputs)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1).unsqueeze(1).repeat(1, self.time_steps, 1)
        
        # Reset states
        self.neuron1.reset()
        self.neuron2.reset()
        
        # Temporal encoding
        encoded = []
        for t in range(self.time_steps):
            x_t = self.encoder(x[:, t])
            x_t = self.bn(x_t)
            x_t = self.dropout(x_t)
            encoded.append(x_t)
        encoded = torch.stack(encoded, dim=1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(encoded)
        
        # Hybrid processing
        spk1 = self.neuron1(lstm_out)
        spk2 = self.neuron2(spk1)
        
        return self.decoder(spk2.mean(dim=1))

# Data Loading
def get_mnist_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    )

# Training Framework
class HybridTrainer:
    def __init__(self, model, device, num_epochs, train_loader, max_lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.total_steps = num_epochs * len(train_loader)
        
        self.optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=1e-4)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=max_lr, 
                                  total_steps=self.total_steps)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for data, targets in loader:
            data, targets = data.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    def validate(self, loader):
        self.model.eval()
        total_loss, correct = 0, 0
        with torch.no_grad():
            for data, targets in loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
        return total_loss / len(loader), 100 * correct / len(loader.dataset)

# Weight Export
def export_weights(model, filename="weights"):
    params = {
        'encoder_weight': model.encoder.weight.detach().cpu().numpy(),
        'encoder_bias': model.encoder.bias.detach().cpu().numpy(),
        'bn_weight': model.bn.weight.detach().cpu().numpy(),
        'bn_bias': model.bn.bias.detach().cpu().numpy(),
        'bn_running_mean': model.bn.running_mean.detach().cpu().numpy(),
        'bn_running_var': model.bn.running_var.detach().cpu().numpy(),
        'decoder_weight': model.decoder.weight.detach().cpu().numpy(),
        'decoder_bias': model.decoder.bias.detach().cpu().numpy(),
        'lstm_weight_ih': model.lstm.weight_ih_l0.detach().cpu().numpy(),
        'lstm_weight_hh': model.lstm.weight_hh_l0.detach().cpu().numpy(),
        'lstm_bias_ih': model.lstm.bias_ih_l0.detach().cpu().numpy(),
        'lstm_bias_hh': model.lstm.bias_hh_l0.detach().cpu().numpy(),
        'neuron1_scale': model.neuron1.scale.detach().cpu().numpy(),
        'neuron1_zero_point': model.neuron1.zero_point.detach().cpu().numpy(),
        'neuron2_scale': model.neuron2.scale.detach().cpu().numpy(),
        'neuron2_zero_point': model.neuron2.zero_point.detach().cpu().numpy(),
    }
    
    # Header file
    with open(f"{filename}.h", "w") as f:
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define INPUT_SIZE {model.encoder.in_features}\n")
        f.write(f"#define HIDDEN_SIZE {model.encoder.out_features}\n")
        f.write(f"#define OUTPUT_SIZE {model.decoder.out_features}\n")
        f.write(f"#define LSTM_HIDDEN_SIZE {model.lstm.hidden_size}\n")
        f.write(f"#define TIME_STEPS {model.time_steps}\n")
        f.write("#define EPSILON 1e-5\n\n")
        
        for name, arr in params.items():
            f.write(f"extern const float {name}[{arr.size}];\n")

    # Source file
    with open(f"{filename}.c", "w") as f:
        for name, arr in params.items():
            f.write(f"const float {name}[] = {{\n")
            f.write(", ".join(map(str, arr.flatten().tolist())) + "\n};\n\n")

# Main Execution
if __name__ == "__main__":
    # Setup environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    
    # Initialize model and trainer
    model = HybridSNNQNN(num_hidden=256, time_steps=20)
    num_epochs = 30
    trainer = HybridTrainer(model, device, num_epochs, train_loader, max_lr=1e-3)
    
    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, acc = trainer.validate(test_loader)
        print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Acc: {acc:.1f}%")
        
        if acc > 95:
            print("Target accuracy reached!")
            break
    
    # Export weights
    export_weights(model)
    print("Weights exported successfully!")
