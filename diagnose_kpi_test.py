# Test script for KPI monitoring
# Simple demonstration of how to use the KPI monitor

import torch
import torch.nn as nn
from diagnoise_kpi_grad import monitor_kpi, reset_monitor

# Create a simple test model
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def test_kpi_monitor():
    """Test the KPI monitoring system"""
    print("Testing KPI Monitor...")
    
    # Reset monitor for clean test
    reset_monitor()
    
    # Create model and test data
    model = TestModel()
    x = torch.randn(32, 10)
    target = torch.randint(0, 5, (32,))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Simulate training epochs
    for epoch in range(1, 11):
        # Forward pass
        output = model(x)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1)
        accuracy = (pred == target).float().mean().item()
        
        # Monitor KPIs (this is the main function to use)
        results = monitor_kpi(
            model=model,
            epoch=epoch,
            save_dir="./test_kpi_logs",
            loss=loss.item(),
            accuracy=accuracy,
            plot_frequency=3  # Generate plots every 3 epochs
        )
        
        print(f"Epoch {epoch}: Loss={loss.item():.4f}, Acc={accuracy:.4f}, "
              f"GradNorm={results['gradient_norm']:.4f}, Mem={results['memory_usage']:.1f}%")
    
    print("Test completed! Check ./test_kpi_logs directory for generated plots and CSV files.")

if __name__ == "__main__":
    test_kpi_monitor()