import torch
from torch.utils.data import DataLoader
import os
from pathlib import Path


from training.datasets import MyDataset
from models.ResNet18 import ResNet18


# -----------------------
# Configuration
# -----------------------

BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

FILES_PATH = Path(Path(os.getcwd()), Path('output/modes'))
print(FILES_PATH)
files = os.listdir(FILES_PATH)
FILES = [file for file in files if file[-3:]=='.h5']

print(FILES)

# -----------------------
# Main
# -----------------------

# def main():

#     # 1. Create train/validation/test split
#     train_indices, val_indices, test_indices = create_split()

#     # 2. Calculate normalization parameters
#     mean, std = calculate_normalization(train_indices)

#     # 3. Create datasets
#     train_dataset = MyDataset(
#         FILES,
#         train_indices,
#         mean=mean,
#         std=std
#     )

#     val_dataset = MyDataset(
#         FILES,
#         val_indices,
#         mean=mean,
#         std=std
#     )

#     test_dataset = MyDataset(
#         FILES,
#         test_indices,
#         mean=mean,
#         std=std
#     )

#     # 4. Create DataLoaders
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True
#     )

#     # 5. Create model
#     model = MyCNN().to(DEVICE)

#     # 6. Loss + optimizer
#     criterion = torch.nn.CrossEntropyLoss()

#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=LEARNING_RATE
#     )

#     # 7. Training
#     best_val_loss = float("inf")

#     for epoch in range(NUM_EPOCHS):

#         train_loss = train_one_epoch(
#             model,
#             train_loader,
#             criterion,
#             optimizer,
#             DEVICE
#         )

#         val_loss, val_accuracy = evaluate(
#             model,
#             val_loader,
#             criterion,
#             DEVICE
#         )

#         print(
#             f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
#             f"train loss: {train_loss:.4f} | "
#             f"val loss: {val_loss:.4f} | "
#             f"val acc: {val_accuracy:.4f}"
#         )

#         # Save best model
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss

#             torch.save(
#                 model.state_dict(),
#                 "best_model.pth"
#             )

#     # 8. Final test evaluation
#     model.load_state_dict(
#         torch.load("best_model.pth")
#     )

#     test_loss, test_accuracy = evaluate(
#         model,
#         test_loader,
#         criterion,
#         DEVICE
#     )

#     print(f"Test loss: {test_loss:.4f}")
#     print(f"Test accuracy: {test_accuracy:.4f}")


# if __name__ == "__main__":
#     main()