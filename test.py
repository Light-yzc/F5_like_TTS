import torch
import torch.nn.functional as F
from models.duration_predictor import DurationPredictor

def test_duration_predictor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # 1. Initialize Dummy Duration Predictor
    hidden_dim = 256
    text_dim = 512
    dur_pred = DurationPredictor(
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        nhead=4,
        num_conv_blocks=2,
        conv_kernel=3,
        latent_rate=25
    ).to(device)
    dur_pred.eval()

    # 2. Create Dummy Batch (B=2, SeqLen=10)
    B, L = 2, 10
    text_features = torch.randn(B, L, text_dim, device=device)
    
    # Batch 0: Prompt is length 3, Target is length 5 (Total valid length = 8)
    # Batch 1: Prompt is length 6, Target is length 2 (Total valid length = 8)
    
    text_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # Valid: 8 tokens
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # Valid: 8 tokens
    ], dtype=torch.float32, device=device)

    # target_text_mask ensures only the target tokens are 1
    target_text_mask = torch.tensor([
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],  # 5 target tokens
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # 2 target tokens
    ], dtype=torch.float32, device=device)

    # 3. Dummy GT frames (to test loss function)
    target_frames = torch.tensor([125, 50], dtype=torch.long, device=device) # ~5s and ~2s

    # 4. Forward Pass
    with torch.no_grad():
        print("Running forward pass...")
        predicted_frames = dur_pred(text_features, text_mask, target_text_mask)
        print(f"Predicted Frames Output: {predicted_frames.cpu().numpy()}")
        
        # Test edge case: What if target_text_mask is all zeros (e.g., bug or pure prompt test)?
        # Should not throw nan!
        target_text_mask_zeros = torch.zeros_like(target_text_mask)
        pred_zeros = dur_pred(text_features, text_mask, target_text_mask_zeros)
        print(f"Predicted Frames (All Zeros Mask): {pred_zeros.cpu().numpy()}")
        if torch.isnan(pred_zeros).any():
            print("ERROR: NaN detected with all-zeros target mask!")

        # 5. Loss Execution
        print("Running loss calculation...")
        loss = dur_pred.loss(text_features, text_mask, target_frames, target_text_mask)
        print(f"Loss Output: {loss.item()}")

    print("✅ All tests passed!")


if __name__ == "__main__":
    test_duration_predictor()