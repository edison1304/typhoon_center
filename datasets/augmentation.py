import torch
import torch.nn as nn
import numpy as np
import random
import torchvision.transforms.functional as TF
from torchvision import transforms
import matplotlib.pyplot as plt
import os

class TyphoonAugmentation:
    def __init__(self, 
                 rotation_prob=1.0,  # 회전 확률 증가
                 rotation_range=(-180, 180),  # 전체 360도 범위로 확장
                 scale_prob=0.0,
                 scale_range=(0.8, 1.2),
                 noise_prob=0.0,
                 noise_level=0.02,
                 debug=False):  # 디버그 모드 추가
        
        self.rotation_prob = rotation_prob
        self.rotation_range = rotation_range
        self.scale_prob = scale_prob
        self.scale_range = scale_range
        self.noise_prob = noise_prob
        self.noise_level = noise_level
        self.debug = debug

    def __call__(self, image):
        """
        Args:
            image: torch.Tensor of shape [C, H, W]
        Returns:
            If debug=True: tuple (final_image, intermediate_images)
            If debug=False: final_image
        """
        # Convert to tensor if numpy array
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
        # 중간 결과를 저장할 리스트
        intermediate_images = [image.clone()]
        
        # 1. Rotation - 태풍의 회전 특성 반영 (360도 전체)
        if random.random() < self.rotation_prob:
            angle = random.uniform(*self.rotation_range)
            image = TF.rotate(image, angle, fill=0, interpolation=TF.InterpolationMode.BILINEAR)
            intermediate_images.append(image.clone())

        # 2. Scale - 태풍 크기 변화 반영
        if random.random() < self.scale_prob:
            scale_factor = random.uniform(*self.scale_range)
            h, w = image.shape[1:]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            image = TF.resize(image, (new_h, new_w))
            # 원래 크기로 다시 조정
            image = TF.center_crop(image, (h, w)) if scale_factor > 1 else \
                   TF.pad(image, [(w-new_w)//2, (h-new_h)//2, 
                                 (w-new_w+1)//2, (h-new_h+1)//2], fill=0)
            intermediate_images.append(image.clone())

        # 3. Gaussian Noise - 센서 노이즈 시뮬레이션
        if random.random() < self.noise_prob:
            noise = torch.randn_like(image) * self.noise_level
            image = image + noise
            image = torch.clamp(image, 0, 1)
            intermediate_images.append(image.clone())

        if self.debug:
            return image, intermediate_images
        return image

    def visualize_steps(self, image, save_path=None):
        """
        시각화를 위한 헬퍼 메소드 - 바로 plot을 보여주고 저장합니다
        Args:
            image: input image
            save_path: 이미지를 저장할 경로 (예: 'path/to/save/aug_steps.png')
        """
        # 디버그 모드 임시 활성화
        original_debug = self.debug
        self.debug = True
        
        image_aug, intermediates = self.__call__(image)
        descriptions = ['Original']
        if len(intermediates) > 1:
            if self.rotation_prob > 0:
                descriptions.append('After Rotation')
            if self.scale_prob > 0:
                descriptions.append('After Scaling')
            if self.noise_prob > 0:
                descriptions.append('After Noise')
        
        # matplotlib 설정
        plt.figure(figsize=(15, 5))
        for idx, (img, desc) in enumerate(zip(intermediates[:len(descriptions)], descriptions)):
            plt.subplot(1, len(descriptions), idx + 1)
            
            # 이미지 전처리
            if torch.is_tensor(img):
                img = img.cpu().detach().numpy()
            
            # 채널 순서 변경 (C,H,W) -> (H,W,C)
            img = np.transpose(img, (1, 2, 0))
            
            # 4채널 이미지를 3채널로 변환 (RGB로 표시)
            if img.shape[-1] == 4:
                # 첫 3개 채널만 사용
                img = img[:, :, :3]
            
            plt.imshow(img)
            plt.title(desc)
            plt.axis('off')
        
        plt.tight_layout()
        
        # 이미지 저장
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Augmentation steps saved to: {save_path}")
        
        plt.show()
        plt.close()  # 메모리 누수 방지를 위해 figure 닫기
        
        # 디버그 모드 복원
        self.debug = original_debug
