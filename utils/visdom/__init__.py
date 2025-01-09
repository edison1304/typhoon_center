import matplotlib.pyplot as plt

def show_images(images, labels, num_images=10):
    """이미지와 라벨을 시각화하는 함수"""
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        img = images[i].cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        plt.imshow(img)
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')
    plt.show()