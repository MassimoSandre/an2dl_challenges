## LIBRARIES IMPORT
if __name__ == "__main__":
    SEED = 42
    # Training configuration
    LEARNING_RATE = 10e-4
    EPOCHS = 500
    PATIENCE = 50

    # Architecture
    HIDDEN_LAYERS = 2        # Hidden layers
    HIDDEN_SIZE = 64        # Neurons per layer

    # Regularisation
    DROPOUT_RATE = 0.35         # Dropout probability
    L1_LAMBDA = 0            # L1 penalty
    L2_LAMBDA = 1e-2            # L2 penalty

    BIDIRECTIONAL = True     # Use bidirectional RNN

    RNN_TYPE = 'GRU'        # 'RNN', 'LSTM', or 'GRU'

    WINDOW_SIZE = 20
    STRIDE = 4
    BATCH_SIZE = 64
    


    import os

    #set environment variables for reproducibility
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"  

    # suppress warnings
    import warnings
    warnings.filterwarnings(action='ignore',category=FutureWarning)
    warnings.filterwarnings(action='ignore',category=Warning)

    # import necessary modules
    import logging
    import random
    import numpy as np

    # set seeds for reproducibility
    np.random.seed(SEED)
    random.seed(SEED)

    # import pytorch 
    import torch
    torch.manual_seed(SEED)
    from torch import nn
    #from torch.utils.tensorboard import SummaryWriter
    from torch.utils.data import TensorDataset, DataLoader
    log_dir = './logs'

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")



    # import other libraries
    import copy
    import shutil
    from itertools import product
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns



    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    def make_loader(ds, batch_size, shuffle, drop_last):
        # Determine optimal number of worker processes for data loading
        cpu_cores = os.cpu_count() or 2
        num_workers = max(2, min(4, cpu_cores))
        #num_workers = 0

        # Create DataLoader with performance optimizations
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True,  # Faster GPU transfer
            pin_memory_device="cuda" if torch.cuda.is_available() else "",
            prefetch_factor=4,  # Load 4 batches ahead
        )



    def recurrent_summary(model, input_size):
        """
        Custom summary function that emulates torchinfo's output while correctly
        counting parameters for RNN/GRU/LSTM layers.

        This function is designed for models whose direct children are
        nn.Linear, nn.RNN, nn.GRU, or nn.LSTM layers.

        Args:
            model (nn.Module): The model to analyze.
            input_size (tuple): Shape of the input tensor (e.g., (seq_len, features)).
        """

        # Dictionary to store output shapes captured by forward hooks
        output_shapes = {}
        # List to track hook handles for later removal
        hooks = []

        def get_hook(name):
            """Factory function to create a forward hook for a specific module."""
            def hook(module, input, output):
                # Handle RNN layer outputs (returns a tuple)
                if isinstance(output, tuple):
                    # output[0]: all hidden states with shape (batch, seq_len, hidden*directions)
                    shape1 = list(output[0].shape)
                    shape1[0] = -1  # Replace batch dimension with -1

                    # output[1]: final hidden state h_n (or tuple (h_n, c_n) for LSTM)
                    if isinstance(output[1], tuple):  # LSTM case: (h_n, c_n)
                        shape2 = list(output[1][0].shape)  # Extract h_n only
                    else:  # RNN/GRU case: h_n only
                        shape2 = list(output[1].shape)

                    # Replace batch dimension (middle position) with -1
                    shape2[1] = -1

                    output_shapes[name] = f"[{shape1}, {shape2}]"

                # Handle standard layer outputs (e.g., Linear)
                else:
                    shape = list(output.shape)
                    shape[0] = -1  # Replace batch dimension with -1
                    output_shapes[name] = f"{shape}"
            return hook

        # 1. Determine the device where model parameters reside
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")  # Fallback for models without parameters

        # 2. Create a dummy input tensor with batch_size=1
        dummy_input = torch.randn(1, *input_size).to(device)

        # 3. Register forward hooks on target layers
        # Iterate through direct children of the model (e.g., self.rnn, self.classifier)
        for name, module in model.named_children():
            if isinstance(module, (nn.Linear, nn.RNN, nn.GRU, nn.LSTM)):
                # Register the hook and store its handle for cleanup
                hook_handle = module.register_forward_hook(get_hook(name))
                hooks.append(hook_handle)

        # 4. Execute a dummy forward pass in evaluation mode
        model.eval()
        with torch.no_grad():
            try:
                model(dummy_input)
            except Exception as e:
                print(f"Error during dummy forward pass: {e}")
                # Clean up hooks even if an error occurs
                for h in hooks:
                    h.remove()
                return

        # 5. Remove all registered hooks
        for h in hooks:
            h.remove()

        # --- 6. Print the summary table ---

        print("-" * 79)
        # Column headers
        print(f"{'Layer (type)':<25} {'Output Shape':<28} {'Param #':<18}")
        print("=" * 79)

        total_params = 0
        total_trainable_params = 0

        # Iterate through modules again to collect and display parameter information
        for name, module in model.named_children():
            if name in output_shapes:
                # Count total and trainable parameters for this module
                module_params = sum(p.numel() for p in module.parameters())
                trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)

                total_params += module_params
                total_trainable_params += trainable_params

                # Format strings for display
                layer_name = f"{name} ({type(module).__name__})"
                output_shape_str = str(output_shapes[name])
                params_str = f"{trainable_params:,}"

                print(f"{layer_name:<25} {output_shape_str:<28} {params_str:<15}")

        print("=" * 79)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {total_trainable_params:,}")
        print(f"Non-trainable params: {total_params - total_trainable_params:,}")
        print("-" * 79)


    class RecurrentClassifier(nn.Module):
        """
        Generic RNN classifier (RNN, LSTM, GRU).
        Uses the last hidden state for classification.
        """
        def __init__(
                self,
                input_size,
                hidden_size,
                num_layers,
                num_classes,
                rnn_type='GRU',        # 'RNN', 'LSTM', or 'GRU'
                bidirectional=False,
                dropout_rate=0.2
                ):
            super().__init__()

            self.rnn_type = rnn_type
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.bidirectional = bidirectional

            # Map string name to PyTorch RNN class
            rnn_map = {
                'RNN': nn.RNN,
                'LSTM': nn.LSTM,
                'GRU': nn.GRU
            }

            if rnn_type not in rnn_map:
                raise ValueError("rnn_type must be 'RNN', 'LSTM', or 'GRU'")

            rnn_module = rnn_map[rnn_type]

            # Dropout is only applied between layers (if num_layers > 1)
            dropout_val = dropout_rate if num_layers > 1 else 0

            # Create the recurrent layer
            self.rnn = rnn_module(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,       # Input shape: (batch, seq_len, features)
                bidirectional=bidirectional,
                dropout=dropout_val
            )

            # Calculate input size for the final classifier
            if self.bidirectional:
                classifier_input_size = hidden_size * 2 # Concat fwd + bwd
            else:
                classifier_input_size = hidden_size

            # Final classification layer
            self.classifier = nn.Linear(classifier_input_size, num_classes)

        def forward(self, x):
            """
            x shape: (batch_size, seq_length, input_size)
            """

            # rnn_out shape: (batch_size, seq_len, hidden_size * num_directions)
            rnn_out, hidden = self.rnn(x)

            # LSTM returns (h_n, c_n), we only need h_n
            if self.rnn_type == 'LSTM':
                hidden = hidden[0]

            # hidden shape: (num_layers * num_directions, batch_size, hidden_size)

            if self.bidirectional:
                # Reshape to (num_layers, 2, batch_size, hidden_size)
                hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)

                # Concat last fwd (hidden[-1, 0, ...]) and bwd (hidden[-1, 1, ...])
                # Final shape: (batch_size, hidden_size * 2)
                hidden_to_classify = torch.cat([hidden[-1, 0, :, :], hidden[-1, 1, :, :]], dim=1)
            else:
                # Take the last layer's hidden state
                # Final shape: (batch_size, hidden_size)
                hidden_to_classify = hidden[-1]

            # Get logits
            logits = self.classifier(hidden_to_classify)
            return logits


    def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, l1_lambda=0, l2_lambda=0):
        """
        Perform one complete training epoch through the entire training dataset.

        Args:
            model (nn.Module): The neural network model to train
            train_loader (DataLoader): PyTorch DataLoader containing training data batches
            criterion (nn.Module): Loss function (e.g., CrossEntropyLoss, MSELoss)
            optimizer (torch.optim): Optimization algorithm (e.g., Adam, SGD)
            scaler (GradScaler): PyTorch's gradient scaler for mixed precision training
            device (torch.device): Computing device ('cuda' for GPU, 'cpu' for CPU)
            l1_lambda (float): Lambda for L1 regularization
            l2_lambda (float): Lambda for L2 regularization

        Returns:
            tuple: (average_loss, f1 score) - Training loss and f1 score for this epoch
        """
        model.train()  # Set model to training mode

        running_loss = 0.0
        all_predictions = []
        all_targets = []

        # Iterate through training batches
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move data to device (GPU/CPU)
            inputs, targets = inputs.to(device), targets.to(device)

            # Clear gradients from previous step
            optimizer.zero_grad(set_to_none=True)

            # Forward pass with mixed precision (if CUDA available)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(inputs)
                loss = criterion(logits, targets)

                # Add L1 and L2 regularization
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm


            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Accumulate metrics
            running_loss += loss.item() * inputs.size(0)
            predictions = logits.argmax(dim=1)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_f1 = f1_score(
            np.concatenate(all_targets),
            np.concatenate(all_predictions),
            average='weighted'
        )

        return epoch_loss, epoch_f1

    def validate_one_epoch(model, val_loader, criterion, device):
        """
        Perform one complete validation epoch through the entire validation dataset.

        Args:
            model (nn.Module): The neural network model to evaluate (must be in eval mode)
            val_loader (DataLoader): PyTorch DataLoader containing validation data batches
            criterion (nn.Module): Loss function used to calculate validation loss
            device (torch.device): Computing device ('cuda' for GPU, 'cpu' for CPU)

        Returns:
            tuple: (average_loss, accuracy) - Validation loss and accuracy for this epoch

        Note:
            This function automatically sets the model to evaluation mode and disables
            gradient computation for efficiency during validation.
        """
        model.eval()  # Set model to evaluation mode

        running_loss = 0.0
        all_predictions = []
        all_targets = []

        # Disable gradient computation for validation
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass with mixed precision (if CUDA available)
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    logits = model(inputs)
                    loss = criterion(logits, targets)

                # Accumulate metrics
                running_loss += loss.item() * inputs.size(0)
                predictions = logits.argmax(dim=1)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        # Calculate epoch metrics
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_accuracy = f1_score(
            np.concatenate(all_targets),
            np.concatenate(all_predictions),
            average='weighted'
        )

        return epoch_loss, epoch_accuracy

    def log_metrics_to_tensorboard(writer, epoch, train_loss, train_f1, val_loss, val_f1, model):
        """
        Log training metrics and model parameters to TensorBoard for visualization.

        Args:
            writer (SummaryWriter): TensorBoard SummaryWriter object for logging
            epoch (int): Current epoch number (used as x-axis in TensorBoard plots)
            train_loss (float): Training loss for this epoch
            train_f1 (float): Training f1 score for this epoch
            val_loss (float): Validation loss for this epoch
            val_f1 (float): Validation f1 score for this epoch
            model (nn.Module): The neural network model (for logging weights/gradients)

        Note:
            This function logs scalar metrics (loss/f1 score) and histograms of model
            parameters and gradients, which helps monitor training progress and detect
            issues like vanishing/exploding gradients.
        """
        # Log scalar metrics
        writer.add_scalar('Loss/Training', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('F1/Training', train_f1, epoch)
        writer.add_scalar('F1/Validation', val_f1, epoch)

        # Log model parameters and gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Check if the tensor is not empty before adding a histogram
                if param.numel() > 0:
                    writer.add_histogram(f'{name}/weights', param.data, epoch)
                if param.grad is not None:
                    # Check if the gradient tensor is not empty before adding a histogram
                    if param.grad.numel() > 0:
                        if param.grad is not None and torch.isfinite(param.grad).all():
                            writer.add_histogram(f'{name}/gradients', param.grad.data, epoch)

    def fit(model, train_loader, val_loader, epochs, criterion, optimizer, scaler, device,
            l1_lambda=0, l2_lambda=0, patience=0, evaluation_metric="val_f1", mode='max',
            restore_best_weights=True, writer=None, verbose=10, experiment_name=""):
        """
        Train the neural network model on the training data and validate on the validation data.

        Args:
            model (nn.Module): The neural network model to train
            train_loader (DataLoader): PyTorch DataLoader containing training data batches
            val_loader (DataLoader): PyTorch DataLoader containing validation data batches
            epochs (int): Number of training epochs
            criterion (nn.Module): Loss function (e.g., CrossEntropyLoss, MSELoss)
            optimizer (torch.optim): Optimization algorithm (e.g., Adam, SGD)
            scaler (GradScaler): PyTorch's gradient scaler for mixed precision training
            device (torch.device): Computing device ('cuda' for GPU, 'cpu' for CPU)
            l1_lambda (float): L1 regularization coefficient (default: 0)
            l2_lambda (float): L2 regularization coefficient (default: 0)
            patience (int): Number of epochs to wait for improvement before early stopping (default: 0)
            evaluation_metric (str): Metric to monitor for early stopping (default: "val_f1")
            mode (str): 'max' for maximizing the metric, 'min' for minimizing (default: 'max')
            restore_best_weights (bool): Whether to restore model weights from best epoch (default: True)
            writer (SummaryWriter, optional): TensorBoard SummaryWriter object for logging (default: None)
            verbose (int, optional): Frequency of printing training progress (default: 10)
            experiment_name (str, optional): Experiment name for saving models (default: "")

        Returns:
            tuple: (model, training_history) - Trained model and metrics history
        """

        # Initialize metrics tracking
        training_history = {
            'train_loss': [], 'val_loss': [],
            'train_f1': [], 'val_f1': []
        }

        # Configure early stopping if patience is set
        if patience > 0:
            patience_counter = 0
            best_metric = float('-inf') if mode == 'max' else float('inf')
            best_epoch = 0

        print(f"Training {epochs} epochs...")

        # Main training loop: iterate through epochs
        for epoch in range(1, epochs + 1):

            # Forward pass through training data, compute gradients, update weights
            train_loss, train_f1 = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, l1_lambda, l2_lambda
            )

            # Evaluate model on validation data without updating weights
            val_loss, val_f1 = validate_one_epoch(
                model, val_loader, criterion, device
            )

            # Store metrics for plotting and analysis
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['train_f1'].append(train_f1)
            training_history['val_f1'].append(val_f1)

            # Write metrics to TensorBoard for visualization
            if writer is not None:
                log_metrics_to_tensorboard(
                    writer, epoch, train_loss, train_f1, val_loss, val_f1, model
                )

            # Print progress every N epochs or on first epoch
            if verbose > 0:
                if epoch % verbose == 0 or epoch == 1:
                    print(f"Epoch {epoch:3d}/{epochs} | "
                        f"Train: Loss={train_loss:.4f}, F1 Score={train_f1:.4f} | "
                        f"Val: Loss={val_loss:.4f}, F1 Score={val_f1:.4f}")

            # Early stopping logic: monitor metric and save best model
            if patience > 0:
                current_metric = training_history[evaluation_metric][-1]
                is_improvement = (current_metric > best_metric) if mode == 'max' else (current_metric < best_metric)

                if is_improvement:
                    best_metric = current_metric
                    best_epoch = epoch
                    torch.save(model.state_dict(), "models/"+experiment_name+'_model.pt')
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered after {epoch} epochs.")
                        break

        # Restore best model weights if early stopping was used
        if restore_best_weights and patience > 0:
            model.load_state_dict(torch.load("models/"+experiment_name+'_model.pt'))
            print(f"Best model restored from epoch {best_epoch} with {evaluation_metric} {best_metric:.4f}")

        # Save final model if no early stopping
        if patience == 0:
            torch.save(model.state_dict(), "models/"+experiment_name+'_model.pt')

        # Close TensorBoard writer
        if writer is not None:
            writer.close()

        return model, training_history




    os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

    print(f'PyTorch version: {torch.__version__} ')
    print(f'Device: {device} ')
    ## DATASET LOADING
    X_train = pd.read_csv('pirate_pain_train.csv')
    y_train = pd.read_csv('pirate_pain_train_labels.csv')

    label_mapping = {'no_pain': 0, 'low_pain': 1, 'high_pain': 2}
    y_train['label'] = y_train['label'].map(label_mapping)

    X_test = pd.read_csv('pirate_pain_test.csv')

    print(f'Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}')
    print(f'Test data shape: {X_test.shape}')


    X_train.info()
    X_train.describe()
    

    unique_train_samples = X_train['sample_index'].unique()
    n_samples = len(unique_train_samples)
    print(f'Number of unique samples in training data: {n_samples}')

    random.seed(SEED)
    random.shuffle(unique_train_samples)

    N_VAL_SAMPLES = int(0.2 * n_samples)
    N_TRAIN_USERS = n_samples - N_VAL_SAMPLES

    train_users = unique_train_samples[:N_TRAIN_USERS]
    val_users = unique_train_samples[N_TRAIN_USERS:]

    df_train = X_train[X_train['sample_index'].isin(train_users)]
    df_val = X_train[X_train['sample_index'].isin(val_users)]
    df_test = X_test

    print(f'Training set shape: {df_train.shape}, Validation set shape: {df_val.shape}, Test set shape: {df_test.shape}')


    # mapping the columns n_legs,n_hands,n_eyes so that if they are "two" then they become 0 else 1 (anything else)
    df_train['n_legs'] = df_train['n_legs'].apply(lambda x: 0 if x == 'two' else 1)
    df_train['n_hands'] = df_train['n_hands'].apply(lambda x: 0 if x == 'two' else 1)
    df_train['n_eyes'] = df_train['n_eyes'].apply(lambda x: 0 if x == 'two' else 1)
    df_val['n_legs'] = df_val['n_legs'].apply(lambda x: 0 if x == 'two' else 1)
    df_val['n_hands'] = df_val['n_hands'].apply(lambda x: 0 if x == 'two' else 1)
    df_val['n_eyes'] = df_val['n_eyes'].apply(lambda x: 0 if x == 'two' else 1)
    df_test['n_legs'] = df_test['n_legs'].apply(lambda x: 0 if x == 'two' else 1)
    df_test['n_hands'] = df_test['n_hands'].apply(lambda x: 0 if x == 'two' else 1)
    df_test['n_eyes'] = df_test['n_eyes'].apply(lambda x: 0 if x == 'two' else 1)   

    scale_columns = [f'joint_{i:02}' for i in range(31)] + ['n_legs','n_hands','n_eyes'] + [f'pain_survey_{i}' for i in range(1,5)] 



    mins = df_train[scale_columns].min()
    maxs = df_train[scale_columns].max()

    # apply min-max scaling (considering case where min=max)
    for col in scale_columns:
        min_val = mins[col]
        max_val = maxs[col]
        if min_val == max_val:
            df_test[col] = 0.0
            df_train[col] = 0.0
            df_val[col] = 0.0
        else:
            df_train[col] = (df_train[col] - min_val) / (max_val - min_val)
            df_val[col] = (df_val[col] - min_val) / (max_val - min_val)
            df_test[col] = (df_test[col] - min_val) / (max_val - min_val)


    for col in scale_columns:
        df_train[col] = df_train[col].astype('float32')
        df_test[col] = df_test[col].astype('float32')
        df_val[col] = df_val[col].astype('float32')        


    # Define a function to build sequences from the dataset
    def build_sequences(df, window=200, stride=200, test=False):
        # Sanity check to ensure the window is divisible by the stride
        assert window % stride == 0

        # Initialise lists to store sequences and their corresponding labels
        dataset = []
        labels = []

        # Iterate over unique IDs in the DataFrame
        for id in df['sample_index'].unique():
            # Extract sensor data for the current ID
            temp = df[df['sample_index'] == id][scale_columns].values

            # Retrieve the activity label for the current ID
            if not test: 
                label = y_train[y_train['sample_index'] == id]['label'].values[0]
            else:
                label = -1  # Placeholder for test data

            # Calculate padding length to ensure full windows
            padding_len = window - len(temp) % window

            # Create zero padding and concatenate with the data
            padding = np.zeros((padding_len, len(scale_columns)), dtype='float32')
            temp = np.concatenate((temp, padding))

            # Build feature windows and associate them with labels
            idx = 0
            while idx + window <= len(temp):
                dataset.append(temp[idx:idx + window])
                labels.append(label)
                idx += stride

        # Convert lists to numpy arrays for further processing
        dataset = np.array(dataset)
        labels = np.array(labels)

        return dataset, labels


    x_train_final,y_train_final = build_sequences(df_train,window=WINDOW_SIZE,stride=STRIDE)
    x_val,y_val = build_sequences(df_val,window=WINDOW_SIZE,stride=STRIDE)
    #x_test,y_test = build_sequences(df_test,window=WINDOW_SIZE,stride=STRIDE,test=True)

    print(f'x_train shape: {x_train_final.shape}, y_train shape: {y_train_final.shape}')
    print(f'x_val shape: {x_val.shape}, y_val shape: {y_val.shape}')
    #print(f'x_test shape: {x_test.shape}')


    input_shape = x_train_final.shape[1:]
    num_classes = len(np.unique(y_train_final))   

    train_ds = TensorDataset(torch.from_numpy(x_train_final).float(), 
                            torch.from_numpy(y_train_final).long())
    val_ds = TensorDataset(torch.from_numpy(x_val).float(), 
                            torch.from_numpy(y_val).long())     

    #test_ds = TensorDataset(torch.from_numpy(x_test).float(),
                           # torch.from_numpy(y_test).long())


    train_loader = make_loader(train_ds, BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = make_loader(val_ds, BATCH_SIZE, shuffle=False, drop_last=False)
    #test_loader = make_loader(test_ds, BATCH_SIZE, shuffle=False, drop_last=False)  

    for xb, yb in train_loader:
        print(f'Input batch shape: {xb.shape}, Labels batch shape: {yb.shape}')
        break
    logs_dir = "logs"
    if not os.path.exists("models"):
        os.makedirs("models")

    # Create model and display architecture with parameter count
    # rnn_model = RecurrentClassifier(
    #     input_size=input_shape[-1], # Pass the number of features
    #     hidden_size=HIDDEN_SIZE,
    #     num_layers=HIDDEN_LAYERS,
    #     num_classes=num_classes,
    #     dropout_rate=DROPOUT_RATE,
    #     bidirectional=False,
    #     rnn_type='RNN'
    #     ).to(device)
    # recurrent_summary(rnn_model, input_size=input_shape)

    # # Set up TensorBoard logging and save model architecture
    # experiment_name = "rnn"
    # writer = SummaryWriter("./"+logs_dir+"/"+experiment_name)
    # x = torch.randn(1, input_shape[0], input_shape[1]).to(device)
    # writer.add_graph(rnn_model, x)

    # # Define optimizer with L2 regularization
    # optimizer = torch.optim.AdamW(rnn_model.parameters(), lr=LEARNING_RATE, weight_decay=L2_LAMBDA)

    # # Enable mixed precision training for GPU acceleration
    # scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    # Create model and display architecture with parameter count
    rnn_model = RecurrentClassifier(
        input_size=input_shape[-1], # Pass the number of features
        hidden_size=HIDDEN_SIZE,
        num_layers=HIDDEN_LAYERS,
        num_classes=num_classes,
        dropout_rate=DROPOUT_RATE,
        bidirectional=BIDIRECTIONAL,
        rnn_type=RNN_TYPE
        ).to(device)
    recurrent_summary(rnn_model, input_size=input_shape)

    

    rnn_model.load_state_dict(torch.load("models/bi_gru_model.pt"))
    rnn_model.eval()

    best_model = rnn_model

    # -----------------------
    # PREDIZIONI SUL TEST SET
    # -----------------------
    # Qui costruiamo le sequenze per il test (senza usare y_train) e facciamo le predizioni
    from collections import Counter
    import csv

    # def build_sequences_test(df, window=WINDOW_SIZE, stride=STRIDE):
    #     """
    #     Costruisce le finestre (same logic as train) ma ritorna:
    #      - dataset: numpy array shape (N_windows, window, n_features)
    #      - indices_per_window: list of sample_index corrispondenti a ciascuna finestra
    #      - unique_indices: list degli sample_index presenti (utile per aggregazione)
    #     """
    #     assert window % stride == 0
    #     dataset = []
    #     indices_per_window = []

    #     for sid in df['sample_index'].unique():
    #         temp = df[df['sample_index'] == sid][scale_columns].values.astype('float32')

    #         padding_len = window - len(temp) % window
    #         if padding_len == window:
    #             padding_len = 0
    #         if padding_len > 0:
    #             padding = np.zeros((padding_len, len(scale_columns)), dtype='float32')
    #             temp = np.concatenate((temp, padding))

    #         idx = 0
    #         while idx + window <= len(temp):
    #             dataset.append(temp[idx:idx + window])
    #             indices_per_window.append(sid)
    #             idx += stride

    #     dataset = np.array(dataset)
    #     return dataset, indices_per_window, list(df['sample_index'].unique())

    # print("Building test windows...")
    # x_test_windows, test_indices_per_window, unique_test_ids = build_sequences_test(df_test, window=WINDOW_SIZE, stride=STRIDE)
    # print(f"Test windows: {x_test_windows.shape[0]} for {len(unique_test_ids)} unique sample_index")

    # # Create DataLoader for test windows
    # test_ds = TensorDataset(torch.from_numpy(x_test_windows).float())
    # test_loader = make_loader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # Predict with best_model
    # best_model.eval()
    # all_preds = []
    # with torch.no_grad():
    #     for batch in test_loader:
    #         inputs = batch[0].to(device)
    #         with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
    #             logits = best_model(inputs)
    #         preds = logits.argmax(dim=1).cpu().numpy()
    #         all_preds.append(preds)
    # all_preds = np.concatenate(all_preds, axis=0)
    # assert len(all_preds) == len(test_indices_per_window)

    # # Aggregate predictions per sample_index (majority vote)
    # preds_by_id = {}
    # for sid, p in zip(test_indices_per_window, all_preds):
    #     preds_by_id.setdefault(sid, []).append(int(p))

    # final_preds_num = {}
    # for sid, plist in preds_by_id.items():
    #     # majority vote; Counter.most_common handles ties by order, fine for deterministic behavior
    #     most_common = Counter(plist).most_common()
    #     final_preds_num[sid] = most_common[0][0]

    # # Map numeric labels back to textual labels
    # inv_label_mapping = {0: 'no_pain', 1: 'low_pain', 2: 'high_pain'}
    # # If la mappatura fosse diversa, adattala qui

    # # Scrivi il CSV con due colonne: sample_index, label (testuale)
    # out_csv = "predictions.csv"
    # with open(out_csv, mode='w', newline='', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['sample_index', 'label'])
    #     for sid in sorted(final_preds_num.keys()):
    #         lbl_text = inv_label_mapping.get(final_preds_num[sid], str(final_preds_num[sid]))
    #         writer.writerow([sid, lbl_text])

    # print(f"Predictions saved to {out_csv} (rows: {len(final_preds_num)})")



    # for col in scale_columns:
    #     df_test[col] = df_test[col].astype('float32')
        
    X_test_tensor = (
        df_test.groupby('sample_index')[scale_columns]
        .apply(lambda x: x.values)
        .tolist()
    )
    test_loader = DataLoader(X_test_tensor, batch_size=32, shuffle=False)

    all_preds = []

    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = best_model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())

    # Convert numeric labels → text labels
    inv_label_map = {0: 'no_pain', 1: 'low_pain', 2: 'high_pain'}
    pred_labels = [inv_label_map[p] for p in all_preds]

    # Build submission DataFrame
    submission = pd.DataFrame({
        "sample_index": sorted(X_test['sample_index'].unique()),
        "label": pred_labels
    })

    # Save to CSV
    submission.to_csv("submission.csv", index=False)

    print("✅ Submission file created: submission.csv")
    submission.head()
