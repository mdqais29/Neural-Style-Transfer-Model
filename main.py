import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing
loader = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

unloader = transforms.ToPILImage()

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def save_image(tensor, path):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(path)

# Gram matrix
def gram_matrix(input):
    batch_size, channels, height, width = input.size()
    features = input.view(batch_size * channels, height * width)
    G = torch.mm(features, features.t())
    return G.div(batch_size * channels * height * width)

# Get features
def get_features(image, model):
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',
        '28': 'conv5_1'
    }

    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def main():
    content_path = input("Enter content image path: ").strip()
    style_path = input("Enter style image path: ").strip()

    content = load_image(content_path)
    style = load_image(style_path)

    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

    content_features = get_features(content, vgg)
    content_features = {k: v.detach() for k, v in content_features.items()}

    style_features = get_features(style, vgg)
    style_features = {k: v.detach() for k, v in style_features.items()}


    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    input_img = content.clone().requires_grad_(True)

    optimizer = optim.Adam([input_img], lr=0.01)

    steps = 300
    style_weight = 1e6
    content_weight = 1

    print("Applying Neural Style Transfer...")

    for step in range(steps):
        input_features = get_features(input_img, vgg)

        content_loss = torch.mean((input_features['conv4_2'] - content_features['conv4_2']) ** 2)

        style_loss = 0
        for layer in style_grams:
            target_feature = input_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_style_loss

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}/{steps}")

    save_image(input_img.detach(), "output_styled.jpg")
    print("✅ Output saved as output_styled.jpg")

if __name__ == "__main__":
    main()