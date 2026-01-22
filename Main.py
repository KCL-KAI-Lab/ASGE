import torch
from Load_Dataset import load_mnist
from VGG8 import VGG8
import matplotlib.pyplot as plt
import time
import copy



if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    train_loader, test_loader,valid_loader, in_c = load_mnist(batch_size=128)

    epochs = 100

    model = VGG8(device=device,epochs=epochs,input_channel=in_c).to(device)

    for opt in model.optimisers:
        for param in opt.param_groups[0]['params']:
            print(param.shape, param.requires_grad)


    train_loss_list = []
    train_acc_list = []
    valid_acc_list = []

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.time()
        train_loss = 0.0
        train_errors = 0
        valid_errors = 0


        for imags, labs in train_loader:
            imags = imags.to(model.device)
            labs = labs.to(model.device)

            batch_loss, batch_errors = model(imags, labs)
            train_loss += batch_loss
            train_errors += batch_errors

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = 100.0 * (1 - train_errors / len(train_loader.dataset))
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)

        for block in model.blocks:
            block.acc_epoch.append(sum(block.acc_batch) / len(block.acc_batch))
            block.acc_batch.clear()

        model.eval()
        with torch.no_grad():
            for imags, labs in valid_loader:
                imags = imags.to(model.device)
                labs = labs.to(model.device)
                batch_errors = model.test(imags, labs)
                valid_errors += batch_errors

            for block in model.blocks:
                block.valid_epoch.append(sum(block.valid_batch) / len(block.valid_batch))
                block.valid_batch.clear()

            avg_valid_acc = 100.0 * (1 - valid_errors / len(valid_loader.dataset))
            valid_acc_list.append(avg_valid_acc)
            end = time.time()
            if avg_valid_acc > best_val_acc:
                best_val_acc = avg_valid_acc
                best_model_state = copy.deepcopy(model.state_dict())

        for scheduler in model.schedulers:
            scheduler.step()

        current_lr = model.classifier_optimiser.param_groups[0]['lr']
        print(
            f"Epoch [{epoch}/{epochs}]  Loss: {avg_train_loss:.4f}  Train_Accuracy: {avg_train_acc:.2f}%  Valid_Accuracy: {avg_valid_acc:.2f}%  Time:{end - start:.2f}s LR:{current_lr}")

        print(f"=== Epoch {epoch} Layer-wise Accuracy ===")
        for i, block in enumerate(model.blocks):
            print(
                f"Layer {i} - {block.acc_epoch[-1]:.4f}")

        print(f"=== Epoch {epoch} Layer-wise Valid ===")
        for i, block in enumerate(model.blocks):
            print(
                f"Block {i} - {block.valid_epoch[-1]:.4f}")

    test_errors = 0

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        for imags,labs in test_loader:
            imags = imags.to(model.device)
            labs = labs.to(model.device)
            batch_errors = model.test(imags, labs)
            test_errors += batch_errors
        avg_test_acc = 100.0 * (1 - test_errors / len(test_loader.dataset))
        with open("Test_Accuracy.txt", "w") as f:
            f.write(f"Test Accuracy: {avg_test_acc:.2f}%\n")


    plt.figure(figsize=(10, 4))
    plt.plot(range(epochs), train_loss_list, label="Train Loss", color='red')
    plt.title("Training Loss vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('Loss.png')

    # 画出准确率曲线
    plt.figure(figsize=(10, 4))
    plt.plot(range(epochs), train_acc_list, label="Train Accuracy", color='blue')
    plt.plot(range(epochs), valid_acc_list, label="Valid Accuracy", color='red')

    max_test_acc = max(valid_acc_list)
    max_epoch = valid_acc_list.index(max_test_acc)
    plt.scatter(max_epoch, max_test_acc, color='red', marker='o', label=f'Max Valid Acc: {max_test_acc:.2f}%')
    plt.text(max_epoch, max_test_acc, f'{max_test_acc:.2f}%', fontsize=10, verticalalignment='bottom')

    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Train & Valid Accuracy vs. Epoch")
    plt.legend()
    plt.savefig("Accuracy.png")