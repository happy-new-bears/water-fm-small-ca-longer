"""
NaN Detection Hook - Add to training to find where NaN first appears
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class NaNDetector:
    """
    Utility to detect NaN/Inf in model forward pass

    Usage:
        detector = NaNDetector()
        detector.register_hooks(model)

        # During training:
        loss, loss_dict = model(batch)
        if detector.has_nan():
            print(f"NaN detected in: {detector.nan_locations}")
            detector.print_report()
    """

    def __init__(self):
        self.nan_locations = []
        self.hooks = []
        self.activations = {}

    def register_hooks(self, model: nn.Module):
        """Register forward hooks on all modules"""
        print("\n" + "="*80)
        print("Registering NaN Detection Hooks")
        print("="*80)

        def make_hook(name):
            def hook(module, input, output):
                # Check input
                if isinstance(input, tuple):
                    for i, inp in enumerate(input):
                        if isinstance(inp, torch.Tensor):
                            if torch.isnan(inp).any():
                                self.nan_locations.append(f"{name}.input[{i}]")
                            if torch.isinf(inp).any():
                                self.nan_locations.append(f"{name}.input[{i}] (Inf)")
                            # Store min/max for diagnosis
                            self.activations[f"{name}.input[{i}]"] = {
                                'min': inp.min().item(),
                                'max': inp.max().item(),
                                'mean': inp.float().mean().item(),
                                'std': inp.float().std().item(),
                            }

                # Check output
                if isinstance(output, torch.Tensor):
                    if torch.isnan(output).any():
                        self.nan_locations.append(f"{name}.output")
                    if torch.isinf(output).any():
                        self.nan_locations.append(f"{name}.output (Inf)")
                    # Store min/max for diagnosis
                    self.activations[f"{name}.output"] = {
                        'min': output.min().item(),
                        'max': output.max().item(),
                        'mean': output.float().mean().item(),
                        'std': output.float().std().item(),
                    }
                elif isinstance(output, tuple):
                    for i, out in enumerate(output):
                        if isinstance(out, torch.Tensor):
                            if torch.isnan(out).any():
                                self.nan_locations.append(f"{name}.output[{i}]")
                            if torch.isinf(out).any():
                                self.nan_locations.append(f"{name}.output[{i}] (Inf)")
                            self.activations[f"{name}.output[{i}]"] = {
                                'min': out.min().item(),
                                'max': out.max().item(),
                                'mean': out.float().mean().item(),
                                'std': out.float().std().item(),
                            }

            return hook

        # Register hooks on all submodules
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(make_hook(name))
                self.hooks.append(hook)

        print(f"✓ Registered {len(self.hooks)} hooks")
        print("="*80 + "\n")

    def has_nan(self) -> bool:
        """Check if NaN was detected"""
        return len(self.nan_locations) > 0

    def reset(self):
        """Reset detection state"""
        self.nan_locations = []
        self.activations = {}

    def print_report(self):
        """Print detailed report of where NaN occurred"""
        print("\n" + "="*80)
        print("NaN DETECTION REPORT")
        print("="*80)

        if not self.has_nan():
            print("✓ No NaN detected!")
            return

        print(f"\n🔴 NaN detected in {len(self.nan_locations)} locations:")
        for loc in self.nan_locations:
            print(f"  - {loc}")

        print("\n" + "="*80)
        print("Activation Statistics (Last Forward Pass)")
        print("="*80)

        # Find problematic activations (very large values)
        fp16_max = 65504.0
        threshold = fp16_max * 0.5  # 50% of FP16 max

        print(f"\n⚠️  Activations close to FP16 overflow (>{threshold:.0f}):")
        dangerous_found = False
        for name, stats in self.activations.items():
            if abs(stats['max']) > threshold or abs(stats['min']) > threshold:
                dangerous_found = True
                print(f"\n  {name}:")
                print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")

        if not dangerous_found:
            print("  None found")

        print("\n" + "="*80)
        print("RECOMMENDATIONS:")
        print("="*80)

        # Check if first NaN is in early layers
        first_nan = self.nan_locations[0]
        if 'embed' in first_nan.lower() or 'input' in first_nan.lower():
            print("\n🔴 NaN in early layers → Data normalization issue!")
            print("   Check: Are input data properly normalized?")
            print("   Check: Are there extreme outliers in data?")
        elif 'attn' in first_nan.lower():
            print("\n🔴 NaN in attention → Attention score overflow!")
            print("   Fix: Add attention clipping or lower temperature")
        elif 'norm' in first_nan.lower():
            print("\n🔴 NaN in normalization layer → Variance too small!")
            print("   Fix: Change eps from 1e-8 to 1e-5 or 1e-6")
        else:
            print("\n🔴 NaN in middle/late layers → Weight explosion!")
            print("   Fix: Add gradient clipping")
            print("   Fix: Lower learning rate")

        print("="*80 + "\n")

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def add_nan_checks_to_training_loop(model, detector: Optional[NaNDetector] = None):
    """
    Add NaN detection to training loop

    Example usage in train_mae.py:

        from utils.nan_detector import NaNDetector, add_nan_checks_to_training_loop

        # After model creation
        detector = NaNDetector()
        detector.register_hooks(model)

        # In training loop
        for batch in train_loader:
            loss, loss_dict = model(batch)

            if detector.has_nan():
                print(f"\\n{'='*80}")
                print(f"NaN DETECTED at Epoch {epoch}, Batch {batch_idx}")
                print(f"{'='*80}")
                detector.print_report()

                # Log batch statistics
                print("\\nBatch Statistics:")
                for key, val in batch.items():
                    if isinstance(val, torch.Tensor):
                        print(f"  {key}: [{val.min():.4f}, {val.max():.4f}]")

                raise RuntimeError("NaN detected - stopping training")

            detector.reset()  # Reset for next batch

            model.backward(loss)
            model.step()
    """
    if detector is None:
        detector = NaNDetector()
        detector.register_hooks(model)
    return detector


if __name__ == '__main__':
    """Test NaN detector"""
    print("Testing NaN Detector...")

    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )

    # Register hooks
    detector = NaNDetector()
    detector.register_hooks(model)

    # Test normal input
    print("\n1. Testing with normal input:")
    x = torch.randn(4, 10)
    y = model(x)
    print(f"   Output: {y}")
    if detector.has_nan():
        detector.print_report()
    else:
        print("   ✓ No NaN detected")
    detector.reset()

    # Test with NaN input
    print("\n2. Testing with NaN input:")
    x_nan = torch.randn(4, 10)
    x_nan[0, 0] = float('nan')
    y_nan = model(x_nan)
    print(f"   Output: {y_nan}")
    if detector.has_nan():
        detector.print_report()
    detector.reset()

    # Test with large input (overflow risk)
    print("\n3. Testing with large input (FP16 overflow risk):")
    x_large = torch.randn(4, 10) * 10000
    y_large = model(x_large)
    print(f"   Output: {y_large}")
    if detector.has_nan():
        detector.print_report()
    else:
        print("   ✓ No NaN detected (but check for large activations)")

    print("\n✓ NaN Detector tests complete")
