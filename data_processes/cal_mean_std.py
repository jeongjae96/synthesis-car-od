import torch
from torchvision import datasets, transforms
from tqdm import tqdm

# 데이터셋의 경로
data_path = 'C:/Users/USER/Desktop/python/competition/Car_OD_data/t'

# 전처리 변환
transform = transforms.Compose([
    # transforms.Resize((224, 224)),  # 이미지 크기 조정
    # transforms.Grayscale(3),
    transforms.ToTensor(),        # 이미지를 텐서로 변환
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지 정규화
])

# 데이터셋 로드
dataset = datasets.ImageFolder(root=data_path, transform=transform)

# DataLoader 생성
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

mean = torch.zeros(3)
std = torch.zeros(3)
N_CHANNELS = 3
for images, labels in tqdm(dataloader):
    for i in range(N_CHANNELS):
        mean[i] += images[:,i,:,:].mean()
        std[i] += images[:,i,:,:].std()

mean.div_(len(dataset))
std.div_(len(dataset))
print(mean, std)
