import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose

class AnimalDataset(Dataset):
    def __init__(self, root, train=True, transform = None):
        self.root = root

        if train:
            mode = "train"
        else:
            mode = "test"

        self.root = os.path.join(root, mode)

        self.categories = [
            'butterfly', 'cat', 'chicken', 'cow', 'dog',
            'elephant', 'horse', 'sheep', 'spider', 'squirrel'
        ]

        self.images_path = []
        self.labels = []
        self.transform = transform

        for i, category in enumerate(self.categories):
            data_files_path = os.path.join(self.root, category)

            for file_name in os.listdir(data_files_path):
                file_path = os.path.join(data_files_path, file_name)
                self.images_path.append(file_path)
                self.labels.append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image) 
        label = self.labels[idx]
        return image, label


if __name__ == '__main__':
    root = "animals"
    transform = Compose([
        Resize((200,200)),
        ToTensor()
    ])
    dataset = AnimalDataset(root=root, train=True, transform = transform)
    # image, label = dataset.__getitem__(1234)
    # print(image)


    training_dataloader = DataLoader(
        dataset = dataset,
        batch_size = 16,
        num_workers =8,
        shuffle = True,
        drop_last = True,
    )

    for images, labels in training_dataloader:
        print(images.shape)
        print(labels)






# if __name__ == '__main__':
#     Chỉ chạy code bên dưới khi gọi trực tiếp file này (python imageforder_manual.py).
#     Trên Windows, bắt buộc có khối này nếu dùng num_workers > 0 trong DataLoader.

#     root = "animals"

#     Compose: gom nhiều bước xử lý ảnh thành 1 pipeline, chạy tuần tự từ trên xuống.
#     Transform chỉ áp dụng khi __getitem__ được gọi (lazy), không biến đổi file gốc trên ổ đĩa.
#     transform = Compose([
#         Resize: đưa mọi ảnh về cùng kích thước (200x200) để batch có shape đồng nhất.
#         Neural network cần tensor cùng shape mới stack thành batch được.
#         Resize((200,200)),
#         ToTensor: PIL Image (H, W, C), pixel 0-255  ->  torch.Tensor (C, H, W), pixel 0.0-1.0
#         ToTensor()
#     ])

#     Tạo dataset: quét folder, lưu danh sách đường dẫn + nhãn số (0-9 theo thứ tự categories).
#     dataset = AnimalDataset(root=root, train=True, transform = transform)
#     Lấy 1 mẫu thủ công: dataset[idx] tương đương dataset.__getitem__(idx)
#     image, label = dataset.__getitem__(1234)
#     print(image)


#     DataLoader: bọc Dataset, tự gom nhiều mẫu thành batch và (tuỳ chọn) load song song.
#     training_dataloader = DataLoader(
#         dataset = dataset,
#         batch_size: số ảnh mỗi lần lấy. Ví dụ 16 -> tensor shape [16, 3, 200, 200]
#         batch_size = 16,
#         num_workers: số process phụ đọc ảnh song song. Tăng tốc I/O; Windows cần if __name__ guard.
#         num_workers =8,
#         shuffle=True: xáo thứ tự mỗi epoch, tránh model học theo thứ tự file trên disk.
#         shuffle = True,
#         drop_last=True: bỏ batch cuối nếu không đủ batch_size (tránh batch lẻ gây lỗi shape).
#         drop_last = True,
#     )

#     Mỗi vòng lặp = 1 batch (không phải 1 ảnh).
#     images: Tensor [batch_size, channels, height, width]
#     labels: Tensor [batch_size] — mỗi phần tử là class index (0=butterfly ... 9=squirrel)
#     for images, labels in training_dataloader:
#         print(images.shape)
#         print(labels)


    