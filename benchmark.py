from data import dataloaders
from model import MiniTransformer
from train import train_model, evaluate
from utils import set_seed, count_parameters, plot_training_curves
import torch


def run_benchmark():
    set_seed(42)
    train_loader, val_loader, test_loader = dataloaders('train.csv', 'validation.csv', 'test.csv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define Model Settings [Alias, PE, Heads, Layers]
    settings = [
        ("Model A", True, 1, 1),
        ("Model B", True, 4, 1),
        ("Model C", False, 4, 1),
        ("Model D", True, 4, 2)
    ]

    results = []

    for name, pe, heads, layers in settings:
        print(f"\n--- Training {name} ---")
        model = MiniTransformer(num_heads=heads, use_pe=pe, num_layers=layers)
        params = count_parameters(model)

        train_losses , val_accs, train_time = train_model(model, train_loader, val_loader, epochs=15)
        if name == "Model B":
            plot_training_curves(train_losses, val_accs, name)

        test_acc = evaluate(model, test_loader, device)

        results.append({
            "Model": name,
            "PE": "Yes" if pe else "No",
            "Heads": heads,
            "Layers": layers,
            "Val Acc": val_accs[-1],
            "Test Acc": test_acc,
            "Train Time": f"{train_time:.2f} min",
            "Params": params
        })


    # Print Final Table
    print("\n" + "=" * 80)
    print(f"{'Model':<10} | {'PE':<4} | {'Heads':<5} | {'Layers':<6} | {'Val Acc':<8} | {'Test Acc':<8} | {'Time'} | {'Params'}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['Model']:<10} | {r['PE']:<4} | {r['Heads']:<5} | {r['Layers']:<6} | {r['Val Acc']:<8.4f} | {r['Test Acc']:<8.4f} | {r['Train Time']} | {r['Params']}")


if __name__ == "__main__":
    run_benchmark()