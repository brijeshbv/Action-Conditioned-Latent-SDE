from rl_zoo3.enjoy import get_encoded_env_samples
import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xs,  ts, actions = get_encoded_env_samples(32, 100, device)
    print(xs.shape, actions.shape)