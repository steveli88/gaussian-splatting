import math

import torch
import torch.nn as nn
import torchvision.models as models


class AppearanceEncoder(nn.Module):
    def __init__(self, backbone="resnet18", output_dim=256, pretrained=True, pooling_window=3):
        super(AppearanceEncoder, self).__init__()

        # Load the backbone (e.g., ResNet, EfficientNet)
        if backbone == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
            feature_dim = self.model.fc.in_features
            self.model = nn.Sequential(*list(self.model.children())[:-2])  # Remove fully connected layers
        elif backbone == "efficientnet_b0":
            self.model = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = self.model.classifier[1].in_features
            self.model = nn.Sequential(*list(self.model.children())[:-2])  # Remove fully connected layers
        else:
            raise ValueError("Unsupported backbone: Choose 'resnet18' or 'efficientnet_b0'.")

        # Adaptive Pooling to handle different input sizes
        self.global_pool = nn.AdaptiveAvgPool2d((pooling_window, pooling_window))

        # Projection head for feature dimensionality reduction
        self.fc = nn.Linear(feature_dim * pooling_window * pooling_window, output_dim)

    def forward(self, x):
        x = self.model(x)  # Feature extraction
        x = self.global_pool(x)  # Global pooling (NxCx1x1)
        x = torch.flatten(x, 1)  # Flatten to (NxC)
        x = self.fc(x)  # Project to output dimension
        return x


class AppearanceTransform(nn.Module):
    def __init__(self, global_encoding_dim, local_encoding_dim, in_dim=3):
        super().__init__()
        self.in_dim = in_dim

        self.mlp_var = nn.Sequential(
            nn.Linear(global_encoding_dim + local_encoding_dim + in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, in_dim),
        )
        
        self.mlp_bias = nn.Sequential(
            nn.Linear(global_encoding_dim + local_encoding_dim + in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, in_dim),
        )

    def forward(self, color, global_encoding, local_encoding, viewdir=None):
        del viewdir  # Viewdirs interface is kept to be compatible with prev. version

        encoding_input = torch.cat((color, global_encoding, local_encoding), dim=-1)
        var = self.mlp_var(encoding_input) # "* 0.01" in wildgaussian why?
        bias = self.mlp_bias(encoding_input)
        return color * var + bias


# Local encoding
def _get_fourier_features(xyz, num_features=3):
    # xyz = torch.from_numpy(xyz).to(dtype=torch.float32)
    xyz = xyz - xyz.mean(dim=0, keepdim=True)
    xyz = xyz / torch.quantile(xyz.abs(), 0.97, dim=0) * 0.5 + 0.5
    freqs = torch.repeat_interleave(
        2**torch.linspace(0, num_features-1, num_features, dtype=xyz.dtype, device=xyz.device), 2)
    offsets = torch.tensor([0, 0.5 * math.pi] * num_features, dtype=xyz.dtype, device=xyz.device)
    feat = xyz[..., None] * freqs[None, None] * 2 * math.pi + offsets[None, None]
    feat = torch.flatten(torch.sin(feat), start_dim=1)
    return feat


if __name__ == "__main__":
    encoding_dim = 64

    # Example usage
    model = AppearanceEncoder(backbone="resnet18", output_dim=encoding_dim, pretrained=False).cuda()
    input_tensor = torch.randn(1, 3, 1224, 624).cuda()  # Batch of 4 images with varying sizes
    features = model(input_tensor)  # (4, 256)
    print(features.shape)

    # Example usage
    appearance_transform = AppearanceTransform(global_encoding_dim=encoding_dim, local_encoding_dim=encoding_dim).cuda()
    color = torch.randn(1000000, 3).cuda()
    global_encoding = torch.randn(1000000, encoding_dim).cuda()
    local_encoding = torch.randn(1000000, encoding_dim).cuda()
    transformed_color = appearance_transform(color, global_encoding, local_encoding)
    print(transformed_color.shape)

    xyz = torch.randn(1000000, 3).cuda()
    position_encoding = _get_fourier_features(xyz, num_features=10)
    print(position_encoding.shape)

