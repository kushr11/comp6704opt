"""
Adversarial Attack Experiment: Comparison of FGSM, PGD, MI-FGSM, and C&W
Evaluated on ImageNet validation set using ResNet50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class NormalizeLayer(nn.Module):
    """Normalize an image from [0,1] range to ImageNet standard normalization"""
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        return (x - self.mean) / self.std


def load_model():
    """Load pretrained ResNet50 model"""
    print("Loading pretrained ResNet50 model...")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = model.to(device)
    model.eval()
    
    # Create full model with normalization (input is in [0,1] range)
    normalize = NormalizeLayer(IMAGENET_MEAN, IMAGENET_STD).to(device)
    full_model = nn.Sequential(normalize, model).to(device)
    full_model.eval()
    
    return full_model


def load_imagenet_data(batch_size=8, num_batches=2):
    """Load ImageNet validation dataset"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    valset = torchvision.datasets.ImageFolder(
        root="imagenet-val",
        transform=transform
    )
    
    # Reproducible sampling
    generator = torch.Generator().manual_seed(42)
    loader = torch.utils.data.DataLoader(
        valset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=generator
    )
    
    print(f"ImageNet validation set size: {len(valset)}")
    print(f"Number of classes: {len(valset.classes)}")
    
    all_images = []
    all_labels = []
    
    for i, (images, labels) in enumerate(loader):
        if i >= num_batches:
            break
        all_images.append(images)
        all_labels.append(labels)
        print(f"  Loaded Batch {i+1}/{num_batches}")
    
    return all_images, all_labels, valset.classes


class AttackResult:
    """Class for storing evaluation results of each attack"""
    def __init__(self):
        self.success_rates = []
        self.convergence_history = []
        self.perturbation_l2 = []
        self.perturbation_linf = []
        self.attack_times = []
        self.successful_examples = []


# ======================== Attack Implementations ========================

def fgsm_attack(model, images, labels, epsilon=8/255):
    """Standard FGSM attack"""
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    grad_sign = images.grad.sign()
    adv_images = images + epsilon * grad_sign
    adv_images = torch.clamp(adv_images, 0, 1)
    
    with torch.no_grad():
        pred = model(adv_images).argmax(dim=1)
        success_rate = (pred != labels).float().mean().item()
    
    return adv_images.detach(), [success_rate]


def pgd_attack(model, images, labels, epsilon=8/255, alpha=2/255, 
               num_iter=10, random_start=True):
    """PGD attack"""
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    if random_start:
        adv_images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(adv_images, 0, 1)
    else:
        adv_images = images.clone()
    
    convergence_history = []
    
    for i in range(num_iter):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        
        with torch.no_grad():
            pred = outputs.argmax(dim=1)
            success_rate = (pred != labels).float().mean().item()
            convergence_history.append(success_rate)
        
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            adv_images = adv_images + alpha * adv_images.grad.sign()
            delta = torch.clamp(adv_images - images, -epsilon, epsilon)
            adv_images = torch.clamp(images + delta, 0, 1)
    
    return adv_images.detach(), convergence_history


def mi_fgsm_attack(model, images, labels, epsilon=8/255, alpha=2/255,
                   num_iter=10, momentum=0.9):
    """Momentum Iterative FGSM attack"""
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    adv_images = images.clone()
    grad_momentum = torch.zeros_like(images)
    
    convergence_history = []
    
    for i in range(num_iter):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        
        with torch.no_grad():
            pred = outputs.argmax(dim=1)
            success_rate = (pred != labels).float().mean().item()
            convergence_history.append(success_rate)
        
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            grad = adv_images.grad
            grad_norm = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
            
            grad_momentum = momentum * grad_momentum + grad_norm
            adv_images = adv_images + alpha * grad_momentum.sign()
            
            delta = torch.clamp(adv_images - images, -epsilon, epsilon)
            adv_images = torch.clamp(images + delta, 0, 1)
    
    return adv_images.detach(), convergence_history


def cw_attack(model, images, labels, c=1.0, kappa=0, num_iter=15, 
              learning_rate=0.01):
    """Simplified C&W L2 attack"""
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    batch_size = images.shape[0]
    
    w = torch.atanh(2 * images.clamp(1e-6, 1-1e-6) - 1)
    w = w.clone().detach().requires_grad_(True)
    
    optimizer = torch.optim.Adam([w], lr=learning_rate)
    
    convergence_history = []
    best_adv = images.clone()
    best_success = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    for i in range(num_iter):
        adv_images = (torch.tanh(w) + 1) / 2
        
        outputs = model(adv_images)
        pred = outputs.argmax(dim=1)
        
        success_mask = (pred != labels)
        success_rate = success_mask.float().mean().item()
        convergence_history.append(success_rate)
        
        improved = success_mask & (~best_success)
        best_adv[improved] = adv_images[improved].detach()
        best_success = best_success | success_mask
        
        one_hot = F.one_hot(labels, num_classes=1000).float()
        real = (one_hot * outputs).sum(dim=1)
        other = ((1 - one_hot) * outputs - one_hot * 1e9).max(dim=1)[0]
        
        f_loss = torch.clamp(real - other + kappa, min=0)
        l2_loss = torch.sum((adv_images - images) ** 2, dim=(1, 2, 3))
        
        loss = l2_loss.mean() + c * f_loss.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    final_adv = (torch.tanh(w) + 1) / 2
    return final_adv.detach(), convergence_history


# ======================== Evaluation ========================

def evaluate_attack(model, clean_images, labels, adv_images):
    """Evaluate attack result"""
    clean_images = clean_images.to(device)
    adv_images = adv_images.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        clean_pred = model(clean_images).argmax(dim=1)
        adv_pred = model(adv_images).argmax(dim=1)
        
        correct_mask = (clean_pred == labels)
        fooled_mask = (adv_pred != labels)
        
        if correct_mask.sum() > 0:
            success_rate = (correct_mask & fooled_mask).float().sum() / correct_mask.float().sum()
        else:
            success_rate = torch.tensor(0.0)
        
        perturbation = (adv_images - clean_images)
        l_inf = perturbation.abs().max().item()
        l2 = torch.sqrt((perturbation ** 2).sum(dim=(1, 2, 3))).mean().item()
    
    return {
        'success_rate': success_rate.item(),
        'l_inf': l_inf,
        'l2': l2,
        'correct_mask': correct_mask.cpu(),
        'fooled_mask': fooled_mask.cpu(),
        'clean_pred': clean_pred.cpu(),
        'adv_pred': adv_pred.cpu()
    }


# ======================== Running Attacks ========================

def run_all_attacks(model, images, labels, epsilon=8/255):
    """Run all four attacks on one batch"""
    alpha = epsilon / 4
    
    attacks = {
        'FGSM': {
            'func': lambda: fgsm_attack(model, images, labels, epsilon),
            'color': '#2196F3',
            'marker': 'o'
        },
        'PGD': {
            'func': lambda: pgd_attack(model, images, labels, epsilon, alpha, num_iter=10),
            'color': '#F44336',
            'marker': 's'
        },
        'MI-FGSM': {
            'func': lambda: mi_fgsm_attack(model, images, labels, epsilon, alpha, num_iter=10),
            'color': '#4CAF50',
            'marker': '^'
        },
        'C&W': {
            'func': lambda: cw_attack(model, images, labels, c=10.0, num_iter=15, learning_rate=0.01),
            'color': '#FF9800',
            'marker': 'd'
        }
    }
    
    results = {}
    
    for name, attack_info in attacks.items():
        print(f"    Running {name} attack...", end=' ')
        
        start_time = time.time()
        adv_images, convergence = attack_info['func']()
        elapsed = time.time() - start_time
        
        eval_result = evaluate_attack(model, images, labels, adv_images)
        
        print(f"Success: {eval_result['success_rate']*100:.1f}%, Time: {elapsed*1000:.1f}ms")
        
        results[name] = {
            'adv_images': adv_images.cpu(),
            'convergence': convergence,
            'eval': eval_result,
            'time': elapsed,
            'color': attack_info['color'],
            'marker': attack_info['marker']
        }
    
    return results


# ======================== Experiment Loop ========================

def run_experiments(batch_size=8, num_batches=2, epsilon=8/255):
    print("=" * 70)
    print("Adversarial Attacks Experiment: FGSM vs PGD vs MI-FGSM vs C&W")
    print("=" * 70)
    
    model = load_model()
    
    print(f"\nLoading ImageNet validation set (batch_size={batch_size}, num_batches={num_batches})...")
    all_images, all_labels, class_names = load_imagenet_data(batch_size, num_batches)
    
    print(f"\nAttack parameter: epsilon = {epsilon*255:.1f}/255")
    
    all_results = {
        'FGSM': AttackResult(),
        'PGD': AttackResult(),
        'MI-FGSM': AttackResult(),
        'C&W': AttackResult()
    }
    attack_configs = {}
    
    for batch_idx in range(num_batches):
        print(f"\n{'='*50}")
        print(f"Batch {batch_idx + 1}/{num_batches}")
        print(f"{'='*50}")
        
        images = all_images[batch_idx]
        labels = all_labels[batch_idx]
        
        with torch.no_grad():
            pred = model(images.to(device)).argmax(dim=1)
            correct = (pred == labels.to(device)).sum().item()
            print(f"  Initial correct predictions: {correct}/{batch_size} ({correct/batch_size*100:.1f}%)")
        
        batch_results = run_all_attacks(model, images, labels, epsilon)
        
        for name, result in batch_results.items():
            attack_configs[name] = {'color': result['color'], 'marker': result['marker']}
            
            all_results[name].success_rates.append(result['eval']['success_rate'])
            all_results[name].convergence_history.append(result['convergence'])
            all_results[name].perturbation_l2.append(result['eval']['l2'])
            all_results[name].perturbation_linf.append(result['eval']['l_inf'])
            all_results[name].attack_times.append(result['time'])
            
            correct_mask = result['eval']['correct_mask']
            fooled_mask = result['eval']['fooled_mask']
            success_indices = torch.where(correct_mask & fooled_mask)[0]
            
            if len(success_indices) > 0 and len(all_results[name].successful_examples) < 1:
                idx = success_indices[0].item()
                all_results[name].successful_examples.append({
                    'original': images[idx],
                    'adversarial': result['adv_images'][idx],
                    'original_label': labels[idx].item(),
                    'original_pred': result['eval']['clean_pred'][idx].item(),
                    'adv_pred': result['eval']['adv_pred'][idx].item(),
                    'class_names': class_names
                })
    
    return all_results, attack_configs, class_names


# Visualization code omitted only due to message length limit.
# I can provide the fully translated visualization section as well.

def main():
    batch_size = 8
    num_batches = 2
    epsilon = 8/255
    
    print("Experiment settings:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Number of batches: {num_batches}")
    print(f"  - Total test samples: {batch_size * num_batches}")
    print(f"  - Epsilon (L∞ constraint): {epsilon*255:.1f}/255")
    
    results, attack_configs, class_names = run_experiments(
        batch_size=batch_size,
        num_batches=num_batches,
        epsilon=epsilon
    )
    
    print("\nExperiment finished!")


if __name__ == "__main__":
    main()
