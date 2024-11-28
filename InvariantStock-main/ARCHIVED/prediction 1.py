import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from Layers import FeatureMask, Predictor, FeatureExtractor, FactorEncoder, FactorDecoder, AlphaLayer, BetaLayer, FatorPrior

# Define a minimal args class with necessary attributes
class Args:
    def __init__(self, device):
        self.device = device

# Set up the device and args
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
args = Args(device=device)

# Load the dataset
dataset = pd.read_pickle('/Users/kevinmwongso/Documents/SMU MQF/qf603_Quantitative_Analysis_of_Financial_Market/PROJECT/InvariantStock-main/data/df_norm.pkl')

# Feature preparation
required_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'change']
for i in range(len(required_features), 20):  # Add dummy columns to make up 20 features
    dataset[f'dummy_feature_{i}'] = 0

# Convert features and labels to tensors
features = dataset[required_features + [f'dummy_feature_{i}' for i in range(len(required_features), 20)]].values
labels = dataset['label'].values  # Only if you want to compare with actuals; not needed for inference

# Extract the date index from the dataset
date_index = dataset.index

# Create PyTorch tensors
features_tensor = torch.tensor(features, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.float32)  # Only for validation, if needed

# Model hyperparameters
input_dim = features_tensor.shape[1]
hidden_dim = 20
factor_dim = 10
batch_size = 64

print('CHECKPOINT1')
# Initialize model components and move to device
feature_mask = FeatureMask(feat_dim=input_dim, hidden_dim=hidden_dim).to(device)
feature_extractor = FeatureExtractor(feat_dim=input_dim, hidden_dim=hidden_dim).to(device)
factor_encoder = FactorEncoder(factor_dims=factor_dim, num_portfolio=input_dim, hidden_dim=hidden_dim).to(device)
factor_decoder = FactorDecoder(AlphaLayer(hidden_dim), BetaLayer(hidden_dim, factor_dim)).to(device)
factor_prior_model = FatorPrior(batch_size=batch_size, hidden_dim=hidden_dim, factor_dim=factor_dim).to(device)
predictor = Predictor(feature_extractor, factor_encoder, factor_decoder, factor_prior_model, args=args).to(device)

# Load model weights and ensure they are on the correct device
feature_mask.load_state_dict(torch.load('/Users/kevinmwongso/Documents/SMU MQF/qf603_Quantitative_Analysis_of_Financial_Market/PROJECT/InvariantStock-main/best_models/best_feat_mask_InvariantStock_1.pt'))
predictor.load_state_dict(torch.load('/Users/kevinmwongso/Documents/SMU MQF/qf603_Quantitative_Analysis_of_Financial_Market/PROJECT/InvariantStock-main/best_models/best_predictor_InvariantStock_1.pt'))
feature_mask.to(device)
predictor.to(device)

print('CHECKPOINT2')
# Set models to evaluation mode
feature_mask.eval()
predictor.eval()

# Prepare DataLoader for test dataset
test_dataset = TensorDataset(features_tensor, labels_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print('CHECKPOINT3')
# Run predictions and calculate ranks
alpha_mu_all = []
beta_all = []

with torch.no_grad():
    for batch_features, _ in test_dataloader:
        # Move batch data to device and add a sequence length dimension
        batch_features = batch_features.unsqueeze(1).to(device)  # Shape: (batch_size, sequence_length=1, num_features)

        # Apply feature mask
        masked_features = feature_mask(batch_features)
        
        # Ensure masked_features has the expected shape
        if masked_features.dim() > 3:
            masked_features = masked_features.view(masked_features.size(0), -1, input_dim)
        
        # Get predictions
        dummy_returns = torch.zeros(batch_features.size(0), device=device)  # Replace with actual returns if needed
        vae_loss, reconstruction_loss, rank_loss, kl_divergence, alpha_mu, beta, factor_mu, factor_sigma, pred_mu, pred_sigma = predictor(masked_features, dummy_returns)
        
        # Append alpha_mu and beta values for each batch
        alpha_mu_all.append(alpha_mu.cpu().numpy())
        beta_all.append(beta.cpu().numpy())

print('CHECKPOINT4')
# Concatenate alpha_mu and beta across all batches
alpha_mu_all = np.concatenate(alpha_mu_all, axis=0)
beta_all = np.concatenate(beta_all, axis=0)

# Create a DataFrame with the date index and save to CSV
alpha_beta_df = pd.DataFrame({
    'date': date_index[:len(alpha_mu_all)],  # Ensure date range matches prediction length
    'alpha_mu': alpha_mu_all.flatten(),  # Flatten in case outputs are multi-dimensional
    'beta': list(beta_all)  # Assuming beta is 2D (for each stock and factor exposure)
})
alpha_beta_df.set_index('date', inplace=True)

print('CHECKPOINT5')
# Save the results if desired
alpha_beta_df.to_csv('alpha_mu_beta_predictions.csv')
print("alpha_mu and beta saved to 'alpha_mu_beta_predictions.csv'")

print('SUCCESS!')
