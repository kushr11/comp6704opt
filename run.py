"""
Adversarial Attack Experiment: Comparison of FGSM, PGD, MI-FGSM, and C&W
Tested on ImageNet validation set using ResNet50
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
    """Normalization layer that normalizes images in [0,1] to ImageNet standard"""
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
    
    # Create full model including normalization (input images in [0,1])
    normalize = NormalizeLayer(IMAGENET_MEAN, IMAGENET_STD).to(device)
    full_model = nn.Sequential(normalize, model).to(device)
    full_model.eval()
    
    return full_model


def load_imagenet_data(batch_size=8, num_batches=2):
    """Load ImageNet validation dataset"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # Convert to [0,1]
    ])
    
    valset = torchvision.datasets.ImageFolder(
        root="imagenet-val",
        transform=transform
    )
    
    # Fixed random seed for reproducibility
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
    
    # Collect a fixed number of batches
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
    """Class to store attack results"""
    def __init__(self):
        self.success_rates = []          # Success rate for each batch
        self.convergence_history = []    # Convergence history per batch
        self.perturbation_l2 = []        # L2 perturbation
        self.perturbation_linf = []      # L∞ perturbation
        self.attack_times = []           # Time cost
        self.successful_examples = []    # Successful examples


# ======================== Attack Implementations ========================

def fgsm_attack(model, images, labels, epsilon=8/255):
    """
    FGSM (Fast Gradient Sign Method) Attack
    Parameters:
        model: Target model
        images: Input images [B, C, H, W], range [0,1]
        labels: Ground-truth labels [B]
        epsilon: Perturbation magnitude
    Returns:
        adv_images: Adversarial examples
        convergence: Convergence history (one value for FGSM)
    """
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
    """
    PGD (Projected Gradient Descent) Attack
    """
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
    """
    MI-FGSM (Momentum Iterative FGSM) Attack
    """
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
    """
    C&W (Carlini & Wagner) L2 Attack - simplified version
    """
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
    """Evaluate attack performance"""
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


def run_all_attacks(model, images, labels, epsilon=8/255):
    """Run all attacks on a batch"""
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
        
        print(f"Success rate: {eval_result['success_rate']*100:.1f}%, Time: {elapsed*1000:.1f}ms")
        
        results[name] = {
            'adv_images': adv_images.cpu(),
            'convergence': convergence,
            'eval': eval_result,
            'time': elapsed,
            'color': attack_info['color'],
            'marker': attack_info['marker']
        }
    
    return results


def run_experiments(batch_size=8, num_batches=2, epsilon=8/255):
    """Run full experiment"""
    print("=" * 70)
    print("Adversarial Attack Experiment: FGSM vs PGD vs MI-FGSM vs C&W")
    print("=" * 70)
    
    model = load_model()
    
    print(f"\nLoading ImageNet validation set (batch_size={batch_size}, num_batches={num_batches})...")
    all_images, all_labels, class_names = load_imagenet_data(batch_size, num_batches)
    
    print(f"\nAttack parameters: epsilon = {epsilon*255:.1f}/255")
    
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
            print(f"  Clean accuracy: {correct}/{batch_size} ({correct/batch_size*100:.1f}%)")
        
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


def visualize_results(results, attack_configs, save_dir='adversarial_results'):
    """Visualize all results"""
    os.makedirs(save_dir, exist_ok=True)
    
    attack_names = list(results.keys())
    colors = [attack_configs[name]['color'] for name in attack_names]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Adversarial Attack Comparison on ImageNet (ResNet50)', fontsize=16, fontweight='bold')
    
    # 1. Success rate comparison
    ax1 = axes[0, 0]
    success_means = [np.mean(results[name].success_rates) * 100 for name in attack_names]
    success_stds = [np.std(results[name].success_rates) * 100 for name in attack_names]
    
    bars = ax1.bar(attack_names, success_means, yerr=success_stds, 
                   color=colors, alpha=0.8, capsize=8, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Attack Success Rate (%)', fontsize=12)
    ax1.set_title('Attack Success Rate Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, mean, std in zip(bars, success_means, success_stds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 2,
                f'{mean:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    # 2. Convergence speed
    ax2 = axes[0, 1]
    
    for name in attack_names:
        all_conv = results[name].convergence_history
        
        max_len = max(len(c) for c in all_conv)
        aligned = []
        for c in all_conv:
            if len(c) < max_len:
                c = c + [c[-1]] * (max_len - len(c))
            aligned.append(c)
        
        mean_conv = np.mean(aligned, axis=0) * 100
        std_conv = np.std(aligned, axis=0) * 100
        
        x = range(1, len(mean_conv) + 1)
        
        if len(mean_conv) == 1:
            ax2.scatter([1], mean_conv, color=attack_configs[name]['color'], 
                       s=150, label=name, marker=attack_configs[name]['marker'],
                       edgecolor='black', linewidth=1.5, zorder=5)
        else:
            ax2.plot(x, mean_conv, color=attack_configs[name]['color'], 
                    linewidth=2.5, label=name, marker=attack_configs[name]['marker'],
                    markersize=8, markeredgecolor='black', markeredgewidth=1)
            ax2.fill_between(x, mean_conv - std_conv, mean_conv + std_conv,
                           color=attack_configs[name]['color'], alpha=0.15)
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Success Rate (%)', fontsize=12)
    ax2.set_title('Convergence Speed Comparison', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    ax2.set_xlim(0.5, 16)
    
    # 3. Time comparison
    ax3 = axes[1, 0]
    times = [np.mean(results[name].attack_times) * 1000 for name in attack_names]
    time_stds = [np.std(results[name].attack_times) * 1000 for name in attack_names]
    
    bars = ax3.bar(attack_names, times, yerr=time_stds, 
                   color=colors, alpha=0.8, capsize=8, edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('Time (ms)', fontsize=12)
    ax3.set_title('Attack Time Comparison', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, t in zip(bars, times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{t:.1f}ms', ha='center', fontsize=11, fontweight='bold')
    
    # 4. Stability comparison
    ax4 = axes[1, 1]
    
    instabilities = []
    for name in attack_names:
        all_conv = results[name].convergence_history
        max_len = max(len(c) for c in all_conv)
        aligned = []
        for c in all_conv:
            if len(c) < max_len:
                c = c + [c[-1]] * (max_len - len(c))
            aligned.append(c)
        std_conv = np.std(aligned, axis=0) * 100
        instabilities.append(np.mean(std_conv))
    
    bars = ax4.bar(attack_names, instabilities, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.2)
    ax4.set_ylabel('Instability (Std across batches, %)', fontsize=12)
    ax4.set_title('Attack Stability Comparison\n(Lower is More Stable)', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    for bar, inst in zip(bars, instabilities):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{inst:.2f}%', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_dir, 'attack_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"\nComparison image saved to: {save_path}")
    
    print("\nSaving attack example images...")
    
    for name in attack_names:
        examples = results[name].successful_examples
        if not examples:
            print(f"  {name}: No successful examples")
            continue
        
        example = examples[0]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{name} Attack Example', fontsize=14, fontweight='bold')
        
        orig_img = example['original'].permute(1, 2, 0).numpy()
        axes[0].imshow(np.clip(orig_img, 0, 1))
        orig_class = example['class_names'][example['original_label']]
        axes[0].set_title(f'Original Image\nTrue Label: {orig_class[:30]}...', fontsize=11)
        axes[0].axis('off')
        
        adv_img = example['adversarial'].permute(1, 2, 0).numpy()
        axes[1].imshow(np.clip(adv_img, 0, 1))
        adv_class = example['class_names'][example['adv_pred']]
        axes[1].set_title(f'Adversarial Image\nPredicted: {adv_class[:30]}...', fontsize=11)
        axes[1].axis('off')
        
        perturbation = adv_img - orig_img
        pert_vis = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min() + 1e-8)
        axes[2].imshow(pert_vis)
        l2_norm = np.sqrt(np.sum(perturbation ** 2))
        linf_norm = np.abs(perturbation).max()
        axes[2].set_title(f'Perturbation (Amplified)\nL2: {l2_norm:.3f}, L∞: {linf_norm:.4f}', fontsize=11)
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{name}_example.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  {name} example saved: {save_path}")
    
    print("\n" + "=" * 80)
    print("Experiment Summary")
    print("=" * 80)
    print(f"\n{'Attack Method':<12} {'Success Rate':<18} {'Avg Time':<15} {'L2 Dist':<12} {'Stability(std)':<12}")
    print("-" * 80)
    
    for name in attack_names:
        avg_success = np.mean(results[name].success_rates) * 100
        std_success = np.std(results[name].success_rates) * 100
        avg_time = np.mean(results[name].attack_times) * 1000
        avg_l2 = np.mean(results[name].perturbation_l2)
        
        all_conv = results[name].convergence_history
        max_len = max(len(c) for c in all_conv)
        aligned = [c + [c[-1]] * (max_len - len(c)) for c in all_conv]
        stability = np.mean(np.std(aligned, axis=0)) * 100
        
        print(f"{name:<12} {avg_success:>6.1f}% ± {std_success:>4.1f}%    "
              f"{avg_time:>8.1f}ms      {avg_l2:>8.4f}    {stability:>6.2f}%")
    
    print("=" * 80)


def main():
    """Main function"""
    BATCH_SIZE = 8
    # NUM_BATCHES = 12
    NUM_BATCHES = 2 # For quicker testing; change to 12 for full experiment
    EPSILON = 8/255
    
    print(f"Experiment configuration:")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Number of batches: {NUM_BATCHES}")
    print(f"  - Total samples: {BATCH_SIZE * NUM_BATCHES}")
    print(f"  - Perturbation bound: {EPSILON*255:.1f}/255")
    
    results, attack_configs, class_names = run_experiments(
        batch_size=BATCH_SIZE,
        num_batches=NUM_BATCHES,
        epsilon=EPSILON
    )
    
    visualize_results(results, attack_configs)
    
    print("\nExperiment completed!")


if __name__ == "__main__":
    main()