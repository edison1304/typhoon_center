from utils.imports import *
from datasets.augmentation import TyphoonAugmentation
import torchvision.transforms.functional as TF

# 데이터셋 클래스 정의
# 위도 경도 채널인 4,5번 밴드는 제외함. 0-3번 밴드만 사용 
# 추후 시도시 변경필요 , loss 도 직접적인 위치를 예측하도록 전달하였음.
# 해야할거  
# augmentation 종류 추가하기 rotation 정도 가능해 보임 
# 

class TyphoonDataset(Dataset):
    def __init__(self, csv_file, root_dir, config, transform=True, crop_size=300, image_size=400, visualize_aug=False):
        self.data = pd.read_csv(csv_file)
        # grade가 4, 5, 6인 데이터만 필터링
        self.data = self.data[self.data['grade'].isin([2,3,4,5])].reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.crop_size = crop_size
        self.image_size = image_size
        self.center = (199, 199)
        self.band_stats = None  # 밴드별 통계 저장용
        self.aug = TyphoonAugmentation(
            rotation_prob=0.7,
            rotation_range=(-30, 30),
            scale_prob=0.3,
            scale_range=(0.8, 1.2),
            noise_prob=0.2
        )
        self.visualize_aug = visualize_aug  # augmentation 시각화 옵션 추가

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 파일 경로 및 이미지 로드
        year = self.data.iloc[idx]['year']
        name = self.data.iloc[idx]['name']
        time_str = self.data.iloc[idx]['time']
        time_obj = pd.to_datetime(time_str)
        filename = time_obj.strftime('%Y%m%d%H%M') + '.tif'
        image_path = os.path.join(self.root_dir, str(year), name, filename)
        
        # 이미지 로드
        image = tiff.imread(image_path)
        
        # 정규화 적용
        image = self.normalize_image(image)
        
        # numpy to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Augmentation 적용 및 시각화
        if self.visualize_aug:
            self.aug.visualize_steps(image)  # augmentation 과정 시각화
        image = self.aug(image)
        
        # 랜덤 크롭 좌표 계산 및 적용 (augmentation 이후)
        max_x = self.image_size - self.crop_size
        max_y = self.image_size - self.crop_size
        crop_x = random.randint(20, max_x-20)
        crop_y = random.randint(20, max_y-20)
        
        image_cropped = TF.crop(image, crop_y, crop_x, self.crop_size, self.crop_size)
        
        # Additional transforms if any        
        label = (crop_x, crop_y)
        grade = self.data.iloc[idx]['grade']
        return {
            'IMAGE': image_cropped,
            'LABEL': torch.tensor(label, dtype=torch.float),
            'GRADE': torch.tensor(grade, dtype=torch.float)
        }

    def analyze_dataset(self):
        """전체 데이터셋의 밴드별 min/max 값을 계산"""
        print("데이터셋 통계 분석 중...")
        mins = [float('inf')] * 6
        maxs = [float('-inf')] * 6
        
        for idx in range(len(self)):
            # CSV에서 각 행의 정보를 가져옴
            year = self.data.iloc[idx]['year']
            name = self.data.iloc[idx]['name']
            time_str = self.data.iloc[idx]['time']
            
            time_obj = pd.to_datetime(time_str)
            filename = time_obj.strftime('%Y%m%d%H%M') + '.tif'
            image_path = os.path.join(self.root_dir, str(year), name, filename)
            
            image = tiff.imread(image_path)
            if image.ndim == 2:
                image = np.expand_dims(image, axis=2)
                
            for band in range(image.shape[2]):
                mins[band] = min(mins[band], np.min(image[:,:,band]))
                maxs[band] = max(maxs[band], np.max(image[:,:,band]))
            
            if idx % 100 == 0:
                print(f"처리 중: {idx}/{len(self)}")
        
        self.band_stats = {'min': mins, 'max': maxs}
        
        print("\n밴드별 통계:")
        for band in range(len(mins)):
            print(f"Band {band}: min = {mins[band]:.2f}, max = {maxs[band]:.2f}")
        
        return self.band_stats

    def normalize_image(self, image):
        """밴드별 min-max 정규화 수행 (0-3 밴드만)"""
        normalized = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
        
        # Band 0-2: min=2000, max=8000으로 고정
        for band in range(3):
            normalized[:,:,band] = (image[:,:,band] - 2000) / (8000 - 2000)
            normalized[:,:,band] = np.clip(normalized[:,:,band], 0, 1)
        
        # Band 3: min=14000, max=16500으로 고정
        normalized[:,:,3] = (image[:,:,3] - 14000) / (16500 - 14000)
        normalized[:,:,3] = np.clip(normalized[:,:,3], 0, 1)
        
        return normalized
