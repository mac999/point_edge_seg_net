# KPI Monitoring for PointEdgeSegNet Training
# Author: AI Assistant
# Date: 2025-09-26
# Purpose: Simple gradient and training KPI monitoring with visualization

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime

class SimpleKPIMonitor:
    def __init__(self):
        self.data = {
            'epochs': [],
            'gradient_norms': [],
            'memory_usage': [],
            'layer_gradients': {}
        }
    
    def update(self, model, epoch, loss=None, accuracy=None):
        """Update KPIs for current epoch"""
        # Calculate gradient norm
        total_norm = 0.0
        layer_grads = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                layer_grads[name] = param_norm.item()
        
        total_norm = total_norm ** 0.5
        
        # Get memory usage
        memory_pct = 0.0
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            memory_pct = allocated * 100
        
        # Store data
        self.data['epochs'].append(epoch)
        self.data['gradient_norms'].append(total_norm)
        self.data['memory_usage'].append(memory_pct)
        
        # Store layer gradients
        for name, grad_norm in layer_grads.items():
            if name not in self.data['layer_gradients']:
                self.data['layer_gradients'][name] = []
            self.data['layer_gradients'][name].append(grad_norm)
        
        # Optional metrics
        if loss is not None:
            if 'loss' not in self.data:
                self.data['loss'] = []
            self.data['loss'].append(loss)
        
        if accuracy is not None:
            if 'accuracy' not in self.data:
                self.data['accuracy'] = []
            self.data['accuracy'].append(accuracy)
        
        return {
            'gradient_norm': total_norm,
            'memory_usage': memory_pct,
            'layer_count': len(layer_grads),
            'alerts': self._check_alerts(total_norm, memory_pct)
        }
    
    def _check_alerts(self, grad_norm, memory_pct):
        """Check for critical conditions"""
        alerts = []
        if grad_norm > 10.0:
            alerts.append(f"Gradient explosion: {grad_norm:.2f}")
        elif grad_norm < 0.001:
            alerts.append(f"Gradient vanishing: {grad_norm:.6f}")
        
        if memory_pct > 90:
            alerts.append(f"High memory usage: {memory_pct:.1f}%")
        
        return alerts
    
    def plot_kpis(self, save_dir, epoch):
        """Generate and save KPI plots"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Gradient norm plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.data['epochs'], self.data['gradient_norms'], 'b-', linewidth=2)
        plt.axhline(y=10.0, color='red', linestyle='--', label='Explosion threshold')
        plt.axhline(y=0.001, color='orange', linestyle='--', label='Vanishing threshold')
        plt.title('Gradient Norm Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'gradient_norm_epoch_{epoch}.png'), dpi=150)
        plt.close()
        
        # 2. Memory usage plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.data['epochs'], self.data['memory_usage'], 'r-', linewidth=2)
        plt.axhline(y=90, color='red', linestyle='--', label='Critical 90%')
        plt.axhline(y=80, color='orange', linestyle='--', label='Warning 80%')
        plt.title('GPU Memory Usage')
        plt.xlabel('Epoch')
        plt.ylabel('Memory Usage (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'memory_usage_epoch_{epoch}.png'), dpi=150)
        plt.close()
        
        # 3. Layer gradient heatmap (if enough data)
        if len(self.data['layer_gradients']) > 0 and len(self.data['epochs']) > 1:
            self._plot_layer_gradients(save_dir, epoch)
        
        # 4. Training metrics (if available)
        if 'loss' in self.data and 'accuracy' in self.data:
            self._plot_training_metrics(save_dir, epoch)
    
    def _plot_layer_gradients(self, save_dir, epoch):
        """Plot layer-wise gradient norms"""
        layer_names = list(self.data['layer_gradients'].keys())
        # print(f"Debug: Found {len(layer_names)} layers: {layer_names[:5]}...")  # Debug info
        
        if not layer_names:
            print("No layer gradients available for plotting")
            return
        
        # More flexible layer filtering for PointEdgeSegNet
        key_layers = []
        # First try specific keywords
        keywords = ['edge_conv', 'conv', 'linear', 'fc', 'classifier', 'weight', 'bias']
        for keyword in keywords:
            matches = [name for name in layer_names if keyword in name.lower()]
            key_layers.extend(matches[:3])  # Take max 3 per keyword
        
        # Remove duplicates and limit to 12 layers
        key_layers = list(dict.fromkeys(key_layers))[:12]
        
        # If still no key layers, take first 10 layers
        if not key_layers:
            key_layers = layer_names[:10]
        
        # print(f"Debug: Selected {len(key_layers)} key layers for plotting")
        
        if not key_layers:
            print("No layers selected for gradient plotting")
            return
        
        plt.figure(figsize=(12, 8))
        plotted_count = 0
        
        for i, layer_name in enumerate(key_layers):
            if layer_name in self.data['layer_gradients']:
                grad_data = self.data['layer_gradients'][layer_name]
                epochs_for_layer = self.data['epochs'][:len(grad_data)]
                
                if len(grad_data) == 0:
                    continue
                
                # Use color map for different layers
                color = plt.cm.tab10(i % 10)
                short_name = layer_name.split('.')[-1] if '.' in layer_name else layer_name
                
                # Handle single epoch case
                if len(epochs_for_layer) == 1:
                    plt.scatter(epochs_for_layer, grad_data, color=color, s=50, 
                              label=short_name[:15], marker='o')
                else:
                    plt.plot(epochs_for_layer, grad_data, color=color, 
                            linewidth=2, label=short_name[:15], marker='o', markersize=3)
                plotted_count += 1
        
        if plotted_count == 0:
            plt.text(0.5, 0.5, 'No gradient data available', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            plt.title('Layer-wise Gradient Norms (No Data)')
        else:
            plt.title(f'Layer-wise Gradient Norms ({plotted_count} layers)')
            plt.yscale('log')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_dir, f'layer_gradients_epoch_{epoch}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # print(f"Layer gradients plot saved: {plot_path} ({plotted_count} layers plotted)")
    
    def _plot_training_metrics(self, save_dir, epoch):
        """Plot training loss and accuracy"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss plot
        epochs_loss = self.data['epochs'][:len(self.data['loss'])]
        ax1.plot(epochs_loss, self.data['loss'], 'b-', linewidth=2, marker='o')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        epochs_acc = self.data['epochs'][:len(self.data['accuracy'])]
        ax2.plot(epochs_acc, self.data['accuracy'], 'g-', linewidth=2, marker='s')
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'training_metrics_epoch_{epoch}.png'), dpi=150)
        plt.close()
    
    def save_csv(self, save_dir, epoch):
        """Save KPI data to CSV"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Main KPI data
        main_data = {
            'epoch': self.data['epochs'],
            'gradient_norm': self.data['gradient_norms'],
            'memory_usage': self.data['memory_usage']
        }
        
        # Add optional metrics
        if 'loss' in self.data:
            main_data['loss'] = self.data['loss'] + [None] * (len(self.data['epochs']) - len(self.data['loss']))
        
        if 'accuracy' in self.data:
            main_data['accuracy'] = self.data['accuracy'] + [None] * (len(self.data['epochs']) - len(self.data['accuracy']))
        
        # Create DataFrame and save
        df = pd.DataFrame(main_data)
        csv_path = os.path.join(save_dir, f'kpi_data_epoch_{epoch}.csv')
        df.to_csv(csv_path, index=False)
        
        # print(f"KPI data saved to: {csv_path}")

# Global monitor instance
_monitor = None

def monitor_kpi(model, epoch, save_dir="./kpi_logs", loss=None, accuracy=None, plot_frequency=1):
    """
    Simple function to monitor KPIs - just pass model and epoch
    
    Args:
        model: PyTorch model
        epoch: Current epoch number
        save_dir: Directory to save plots and CSV
        loss: Optional training loss
        accuracy: Optional training accuracy
        plot_frequency: Generate plots every N epochs
    
    Returns:
        Dictionary with current KPI values and alerts
    """
    global _monitor
    
    # Initialize monitor on first use
    if _monitor is None:
        _monitor = SimpleKPIMonitor()
        print(f"KPI Monitor initialized. Saving to: {save_dir}")
    
    # Update KPIs
    results = _monitor.update(model, epoch, loss, accuracy)
    
    # Generate plots periodically
    if epoch % plot_frequency == 0 or epoch == 1:
        _monitor.plot_kpis(save_dir, epoch)
        _monitor.save_csv(save_dir, epoch)
        # print(f"KPI plots and data saved for epoch {epoch}")
    
    # Print alerts
    if results['alerts']:
        print("KPI ALERTS:")
        for alert in results['alerts']:
            print(f"  - {alert}")
    
    return results

def reset_monitor():
    """Reset the global monitor (useful for new training sessions)"""
    global _monitor
    _monitor = None
    print("KPI Monitor reset")

# Example usage
if __name__ == "__main__":
    print("Simple KPI Monitor for PointEdgeSegNet")
    print("Usage in training loop:")
    print("  results = monitor_kpis(model, epoch, loss=train_loss, accuracy=train_acc)")
    print("  # Generates plots every 5 epochs by default")
