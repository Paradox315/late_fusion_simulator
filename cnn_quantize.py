import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import os
import copy  # To keep a copy of the original model


# --- 1. Model Definition ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Feature extractor: Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool
        self.features = nn.Sequential(
            # Layer 0: Conv2d
            nn.Conv2d(
                in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1
            ),
            # Layer 1: ReLU
            nn.ReLU(),
            # Layer 2: MaxPool2d
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 3: Conv2d
            nn.Conv2d(
                in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            # Layer 4: ReLU
            nn.ReLU(),
            # Layer 5: MaxPool2d
            nn.MaxPool2d(kernel_size=2, stride=2)
            # Output shape: (batch_size, 16, 7, 7) for MNIST 28x28 input
        )
        # Classifier: Flatten -> Linear -> ReLU -> Linear
        self.classifier = nn.Sequential(
            # Layer 6: Flatten
            nn.Flatten(),
            # Layer 7: Linear
            nn.Linear(16 * 7 * 7, 64),  # 16 channels * 7x7 feature map
            # Layer 8: ReLU
            nn.ReLU(),
            # Layer 9: Linear (Output Layer)
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    # --- Function to fuse modules ---
    # It's good practice to define fusion patterns specific to the model
    def fuse_model(self, is_qat=False):
        # Fuse Conv -> ReLU
        # Fuse Conv -> ReLU -> MaxPool is not a standard direct fusion,
        # but fusing Conv-ReLU is common before quantization.
        # We fuse based on module type sequence in self.features and self.classifier
        modules_to_fuse = []

        # Iterate through sequential blocks
        for name, module_block in [
            ("features", self.features),
            ("classifier", self.classifier),
        ]:
            # Use a list of module names for potential fusion
            module_list = list(module_block.named_children())
            for i in range(len(module_list) - 1):
                current_module_name, current_module = module_list[i]
                next_module_name, next_module = module_list[i + 1]

                # Fuse Conv2d -> ReLU
                if isinstance(current_module, nn.Conv2d) and isinstance(
                    next_module, nn.ReLU
                ):
                    modules_to_fuse.append(
                        [f"{name}.{current_module_name}", f"{name}.{next_module_name}"]
                    )

                # Fuse Linear -> ReLU
                elif isinstance(current_module, nn.Linear) and isinstance(
                    next_module, nn.ReLU
                ):
                    modules_to_fuse.append(
                        [f"{name}.{current_module_name}", f"{name}.{next_module_name}"]
                    )

                # Add other fusion patterns if needed (e.g., Conv-BN-ReLU)
                # Note: This simple model doesn't have BatchNorm

        print(f"Modules to fuse: {modules_to_fuse}")
        if modules_to_fuse:
            torch.quantization.fuse_modules(self, modules_to_fuse, inplace=True)
            print("Fusion complete.")
        else:
            print("No modules to fuse based on defined patterns.")


# --- Helper Functions ---
def train_model(model, train_loader, criterion, optimizer, num_epochs=1, device="cpu"):
    """Basic training loop"""
    model.train()
    model.to(device)
    print(f"Starting training for {num_epochs} epochs on {device}...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:  # Print every 100 batches
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}"
                )
                running_loss = 0.0
    print("Training finished.")
    model.to("cpu")  # Move back to CPU for quantization steps


def evaluate_model(model, test_loader, description="Model", device="cpu"):
    """Evaluate model accuracy"""
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy of the {description} on the test images: {accuracy:.2f} %")
    model.to("cpu")  # Ensure model is on CPU after evaluation
    return accuracy


def print_size_of_model(model, label=""):
    """Prints the size of the model"""
    torch.save(model.state_dict(), "temp_model_state.p")
    size = os.path.getsize("temp_model_state.p") / 1e6  # Size in MB
    print(f"Size ({label}): {size:.2f} MB")
    os.remove("temp_model_state.p")
    return size


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pretrained = True  # Set to False to train from scratch
    pretrained_model_path = "simple_cnn_mnist_float.pth"
    quantized_model_path = "simple_cnn_mnist_quant_qnnpack.pt"
    num_calibration_batches = 50  # Number of batches for calibration
    num_epochs_train = 1  # Only train for 1 epoch if not using pre-trained

    # --- Dataset and Loaders ---
    print("Preparing dataset (MNIST)...")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST specific normalization
        ]
    )

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Create a calibration loader from a subset of the training data
    calibration_indices = torch.randperm(len(train_dataset))[
        : num_calibration_batches * train_loader.batch_size
    ]
    calibration_subset = Subset(train_dataset, calibration_indices)
    calibration_loader = DataLoader(
        calibration_subset, batch_size=train_loader.batch_size
    )  # Use same batch size
    print(f"Using {len(calibration_subset)} samples for calibration.")

    # --- 2. Load or Train FP32 Model ---
    float_model = SimpleCNN(num_classes=10)

    if use_pretrained and os.path.exists(pretrained_model_path):
        print(f"Loading pre-trained FP32 model from {pretrained_model_path}")
        float_model.load_state_dict(
            torch.load(pretrained_model_path, map_location="cpu")
        )
    else:
        print("Training FP32 model...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(float_model.parameters(), lr=0.001)
        train_model(
            float_model,
            train_loader,
            criterion,
            optimizer,
            num_epochs=num_epochs_train,
            device=device,
        )
        print(f"Saving trained FP32 model to {pretrained_model_path}")
        torch.save(float_model.state_dict(), pretrained_model_path)

    # Keep a copy for comparison
    float_model_copy = copy.deepcopy(float_model)
    float_model.eval()  # Ensure model is in eval mode

    # --- Evaluate FP32 model ---
    print("\n--- Evaluating FP32 Model ---")
    fp32_accuracy = evaluate_model(
        float_model, test_loader, "FP32 Model", device="cpu"
    )  # Evaluate on CPU like quantized
    fp32_size = print_size_of_model(float_model, "FP32 Model")

    # --- 3. Static Quantization with QNNPACK ---
    print("\n--- Starting Static Quantization (QNNPACK Backend) ---")

    # Ensure the model is on CPU and in eval mode
    quant_model = copy.deepcopy(float_model_copy)  # Work on a copy
    quant_model.eval()
    quant_model.cpu()

    # --- Step 3.1: Specify Backend ---
    # IMPORTANT: Set the backend *before* fusion and preparation
    print("Setting quantization backend to QNNPACK...")
    torch.backends.quantized.engine = "qnnpack"

    # --- Step 3.2: Fuse Modules ---
    print("Fusing modules...")
    quant_model.fuse_model()  # Call the model-specific fusion method

    # --- Step 3.3: Specify Quantization Configuration ---
    # Use the default qconfig for QNNPACK (usually appropriate for ARM)
    # Typically uses MinMaxObserver for activations and PerChannelMinMaxObserver for weights (INT8)
    print("Applying QNNPACK default qconfig...")
    quant_model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
    print(f"Model qconfig: {quant_model.qconfig}")

    # --- Step 3.4: Prepare the Model for Calibration ---
    # Inserts observers to collect activation statistics
    print("Preparing model for static quantization (inserting observers)...")
    # Important: Use inplace=False first to debug if needed, then switch to True
    prepared_model = torch.quantization.prepare(quant_model, inplace=False)
    prepared_model.eval()  # Ensure eval mode after prepare

    # --- Step 3.5: Calibrate the Model ---
    print(f"Running calibration with {num_calibration_batches} batches...")
    with torch.no_grad():
        for i, (calib_data, _) in enumerate(calibration_loader):
            prepared_model(calib_data)
            if (i + 1) % 10 == 0:  # Print progress
                print(f"  Calibration batch {i+1}/{len(calibration_loader)}")
            if i + 1 >= num_calibration_batches:  # Limit calibration batches if needed
                break
    print("Calibration finished.")

    # --- Step 3.6: Convert the Model to Quantized Format ---
    # Removes observers, calculates scale/zero-point, replaces modules with quantized versions
    print("Converting model to quantized version...")
    quantized_model = torch.quantization.convert(prepared_model, inplace=False)
    quantized_model.eval()  # Ensure eval mode after conversion
    print("Model conversion complete.")

    # --- 4. Evaluate Quantized Model ---
    print("\n--- Evaluating Quantized (INT8) Model ---")
    # Ensure evaluation is on CPU as QNNPACK is a CPU backend
    int8_accuracy = evaluate_model(
        quantized_model, test_loader, "Quantized INT8 Model", device="cpu"
    )

    # --- 5. Compare Model Sizes ---
    print("\n--- Comparing Model Sizes ---")
    print_size_of_model(float_model_copy, "Original FP32 Model")
    int8_size = print_size_of_model(quantized_model, "Quantized INT8 Model")
    print(f"Size reduction: {((fp32_size - int8_size) / fp32_size * 100):.2f}%")

    # --- 6. Save Quantized Model as TorchScript ---
    print(f"\n--- Saving Quantized Model to TorchScript ({quantized_model_path}) ---")
    # Scripting is required for loading in C++ environments (like PyTorch Mobile)
    try:
        scripted_quantized_model = torch.jit.script(quantized_model)
        scripted_quantized_model.save(quantized_model_path)
        print(f"Quantized model successfully saved to {quantized_model_path}")
    except Exception as e:
        print(f"Error saving scripted model: {e}")
        print("Trying to save state_dict instead (not directly deployable to mobile).")
        torch.save(quantized_model.state_dict(), "quantized_cnn_qnnpack_statedict.pth")

    print("\nQuantization process finished.")


# --- 7. Example Output Simulation ---
# This section shows what the output might look like when you run the script.
# (Actual numbers will vary based on training, calibration data, etc.)
"""
--- Example Output ---
Preparing dataset (MNIST)...
Using 3200 samples for calibration.
Loading pre-trained FP32 model from simple_cnn_mnist_float.pth

--- Evaluating FP32 Model ---
Accuracy of the FP32 Model on the test images: 97.85 %
Size (FP32 Model): 0.21 MB

--- Starting Static Quantization (QNNPACK Backend) ---
Setting quantization backend to QNNPACK...
Fusing modules...
Modules to fuse: [['features.0', 'features.1'], ['features.3', 'features.4'], ['classifier.1', 'classifier.2']]
Fusion complete.
Applying QNNPACK default qconfig...
Model qconfig: QConfig(activation=functools.partial(<class 'torch.quantization.observer.MinMaxObserver'>, dtype=torch.quint8, qscheme=torch.per_tensor_affine), weight=functools.partial(<class 'torch.quantization.observer.PerChannelMinMaxObserver'>, dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
Preparing model for static quantization (inserting observers)...
Running calibration with 50 batches...
  Calibration batch 10/50
  Calibration batch 20/50
  Calibration batch 30/50
  Calibration batch 40/50
  Calibration batch 50/50
Calibration finished.
Converting model to quantized version...
Model conversion complete.

--- Evaluating Quantized (INT8) Model ---
Accuracy of the Quantized INT8 Model on the test images: 97.52 %

--- Comparing Model Sizes ---
Size (Original FP32 Model): 0.21 MB
Size (Quantized INT8 Model): 0.06 MB
Size reduction: 71.43%

--- Saving Quantized Model to TorchScript (simple_cnn_mnist_quant_qnnpack.pt) ---
Quantized model successfully saved to simple_cnn_mnist_quant_qnnpack.pt

Quantization process finished.
"""
