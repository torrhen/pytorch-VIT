import torch

def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    accuracy_fn,
    device: torch.device):
  
  epoch_loss, epoch_accuracy = 0.0, 0.0
  n_batches = len(dataloader)

  model.train()
  for batch_idx, (inputs, targets) in enumerate(dataloader):
    inputs, targets = inputs.to(device), targets.to(device)
    # stop accumulating gradients between batch
    optimizer.zero_grad()
    outputs = model(inputs).to(device)
    # calculate loss
    loss = loss_fn(outputs, targets)
    epoch_loss += loss.item()
    # calculate accuracy
    accuracy = accuracy_fn(outputs, targets)
    epoch_accuracy += accuracy.item()
    # backpropagation
    loss.backward()
    # gradient descent update
    optimizer.step()

    if (batch_idx > 0) and (batch_idx % 10 == 0):
      print(f"| Batch: {batch_idx}\t| Loss: {epoch_loss / (batch_idx + 1):.4f} | Accuracy: {epoch_accuracy / (batch_idx + 1):.4f} |")

  epoch_loss /= n_batches
  epoch_accuracy /= n_batches

  return epoch_loss, epoch_accuracy


def test_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn,
    device: torch.device):
  
  epoch_loss, epoch_accuracy = 0.0, 0.0
  n_batches = len(dataloader)

  # inference mode
  model.eval()
  # stop tracking gradients
  with torch.no_grad():
    for (inputs, targets) in dataloader:
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = model(inputs).to(device)
      # calculate loss
      loss = loss_fn(outputs, targets)
      epoch_loss += loss.item()
      # calculate accuracy
      accuracy = accuracy_fn(outputs, targets)
      epoch_accuracy += accuracy.item()

  epoch_loss /= n_batches
  epoch_accuracy /= n_batches

  return epoch_loss, epoch_accuracy


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    accuracy_fn,
    device: torch.device,
    epochs: int):
  
  results = {
    'train_loss' : [],
    'train_accuracy' : [],
    'val_loss' : [],
    'val_accuracy' : []
  }

  for epoch in range(epochs):
    train_results = train_epoch(
      model=model,
      dataloader=train_dataloader,
      optimizer=optimizer,
      loss_fn=loss_fn,
      accuracy_fn=accuracy_fn,
      device=device,
    )

    val_results = test_epoch(
      model=model,
      dataloader=val_dataloader,
      loss_fn=loss_fn,
      accuracy_fn=accuracy_fn,
      device=device
    )

    results['train_loss'].append(train_results[0])
    results['train_accuracy'].append(train_results[1])
    results['val_loss'].append(val_results[0])
    results['val_accuracy'].append(val_results[1])

    print(f"\nEpoch: {epoch + 1}")
    print("----------")
    print("Train Results")
    print("----------")
    print(f"| Loss: {results['train_loss'][epoch]:.4f} | Accuracy: {results['train_accuracy'][epoch]:.4f} |\n")
    print("Validation Results")
    print("----------")
    print(f"| Loss: {results['val_loss'][epoch]:.4f} | Accuracy: {results['val_accuracy'][epoch]:.4f} |\n")

  return results