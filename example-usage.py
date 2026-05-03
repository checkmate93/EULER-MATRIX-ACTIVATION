import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# Ρύθμιση Συσκευής
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. ADAPTIVE EULER ACTIVATION (The Pro Version)
# ==========================================
class AdaptiveEulerActivation(nn.Module):
    def __init__(self, num_features=128):
        super(AdaptiveEulerActivation, self).__init__()
        # Vectorized Phase: Κάθε νευρώνας μαθαίνει τη δική του γωνία περιστροφής
        self.theta = nn.Parameter(torch.randn(num_features) * 0.1 + 0.45)
        # Amplitude Scaling: Μαθαίνει να ενισχύει ή να αποσβένει το σήμα
        self.alpha = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        cos_t = torch.cos(self.theta)
        sin_t = torch.sin(self.theta)
        # Η "μαγεία" της μιγαδικής περιστροφής ανά κανάλι
        return self.alpha * (cos_t * x - sin_t * torch.tanh(x))

# ==========================================
# 2. ADAPTIVE LLM ARCHITECTURE (30 LAYERS)
# ==========================================
class AdaptiveLLMNet(nn.Module):
    def __init__(self, mode='euler'):
        super(AdaptiveLLMNet, self).__init__()
        
        layers = [nn.Flatten(), nn.Linear(28*28, 128), nn.LayerNorm(128)]
        
        # Προσθήκη 29 επιπέδων (Σύνολο 30 μαζί με το input layer)
        for i in range(29):
            act = AdaptiveEulerActivation(128) if mode == 'euler' else nn.GELU()
            layers.append(act)
            if i < 28: # Μέχρι το προτελευταίο επίπεδο
                layers.append(nn.Linear(128, 128))
                layers.append(nn.LayerNorm(128))
            
        layers.append(nn.Linear(128, 10))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ==========================================
# 3. TRAINING ENGINE
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000, shuffle=False)

def train_adaptive_test(mode, epochs=5):
    model = AdaptiveLLMNet(mode=mode).to(device)
    # AdamW: Θα βελτιστοποιήσει και τις παραμέτρους theta/alpha της Euler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4) 
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n🚀 Εκκίνηση PRO Test: {mode.upper()} (Adaptive & Deep)")
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f}")
    return model

# ==========================================
# 4. FINAL PROOF OF CONCEPT (Evaluation)
# ==========================================
def run_final_benchmark(euler_model, gelu_model):
    def get_accuracy(model):
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        return 100. * correct / len(test_loader.dataset)

    print("\n" + "="*40)
    print("🏆 FINAL PROOF OF CONCEPT RESULTS")
    print("="*40)
    
    e_acc = get_accuracy(euler_model)
    g_acc = get_accuracy(gelu_model)
    
    print(f"Adaptive Euler Accuracy: {e_acc:.2f}%")
    print(f"Standard GELU Accuracy:   {g_acc:.2f}%")
    print("-" * 40)
    
    if e_acc > g_acc:
        print(f"✨ SUCCESS: Η Adaptive Euler ξεπέρασε τη GELU κατά {e_acc - g_acc:.2f}%!")
    else:
        print(f"📊 Σχεδόν ισοπαλία: Διαφορά μόλις {abs(e_acc - g_acc):.2f}%")
    print("="*40)

# Εκτέλεση
euler_pro = train_adaptive_test('euler')
gelu_pro = train_adaptive_test('gelu')
run_final_benchmark(euler_pro, gelu_pro)