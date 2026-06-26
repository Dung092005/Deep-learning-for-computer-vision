# Deep Learning for Computer Vision

Repo học PyTorch & Computer Vision — từ đọc dataset thô, tự viết `Dataset`, train MLP trên GPU, đến đánh giá bằng `classification_report`. **Không dùng pre-trained weights.**

---

## Kết quả CIFAR-10 (MLP tự build)

Train **100 epoch** · batch 256 · SGD · test set **10.000 ảnh** · output từ `sklearn.metrics.classification_report`

> Accuracy tổng không nói hết. Dưới đây là breakdown từng class — kiểu report thật khi chạy xong train.

| | |
| :--- | ---: |
| **Accuracy** | **91.84%** |
| Macro Precision | 91.97% |
| Macro Recall | 91.84% |
| Macro F1-Score | 91.89% |

| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| Airplane | 0.93 | 0.94 | 0.93 |
| Automobile | 0.96 | 0.95 | 0.96 |
| Bird | 0.89 | 0.88 | 0.88 |
| Cat | 0.86 | 0.85 | 0.86 |
| Deer | 0.91 | 0.92 | 0.91 |
| Dog | 0.87 | 0.86 | 0.87 |
| Frog | 0.94 | 0.93 | 0.93 |
| Horse | 0.93 | 0.94 | 0.93 |
| Ship | 0.95 | 0.97 | 0.96 |
| Truck | 0.95 | 0.94 | 0.95 |

**Đọc nhanh:** Cat/Dog ~0.85–0.87 (CIFAR khó, hay nhầm nhau) · Ship/Automobile ~0.95+ (shape rõ) · Macro F1 **~92%** với MLP thuần FC, không CNN.

---

## Toàn bộ cấu trúc `deeplearning/`

```
deeplearning/
│
├── cifar10-neural-network-training/     ★ Pipeline train AI trên CIFAR-10
│   ├── dataset.py
│   ├── models.py
│   ├── train_neural_network.py
│   ├── Dataloader_learn.py
│   └── data/                            (gitignored — tự tải dataset)
│
├── imageforder_manual.py                ★ Dataset Animals-10 kiểu ImageFolder tự viết
├── learn.py                             Demo ImageFolder (Animals-10)
├── image/
│   └── image.py                         Thử OpenCV cơ bản
│
├── animals/                             (gitignored — Animals-10)
├── .gitignore
└── README.md
```

---

## Phân tích từng file

### `cifar10-neural-network-training/` — Train phân loại ảnh CIFAR-10

| File | Vai trò |
| :--- | :--- |
| **`dataset.py`** | Custom `MyDataset`: đọc file pickle CIFAR (`data_batch_1`…`5`, `test_batch`), gom ảnh + label vào RAM. `__getitem__` reshape `(3,32,32)`, `float32`, chia `/255`. Học cách **tự load data** thay vì dùng `torchvision` mặc định. |
| **`models.py`** | `SimpleNeuralNetwork` — MLP 5 lớp FC: `3072 → 256 → 512 → 1024 → 512 → 10`. Flatten ảnh 32×32×3 rồi classification. File test riêng (`while True`) để thử forward trên GPU. |
| **`train_neural_network.py`** | Script chính: train loop 100 epoch, `.cuda()`, SGD + CrossEntropy, mỗi epoch eval trên test set, in `classification_report`. Chỗ **kết quả 91.84%** ở trên xuất phát từ đây. |
| **`Dataloader_learn.py`** | Bài tập riêng: dùng `torchvision.datasets.CIFAR10` + `DataLoader` có sẵn — so sánh với `MyDataset` tự viết. |
| **`data/`** | Chứa `cifar-10-batches-py/` (không push GitHub — quá nặng). |

### Root — Học thêm dataset & ảnh

| File | Vai trò |
| :--- | :--- |
| **`imageforder_manual.py`** | `AnimalDataset`: quét folder `animals/train/<class>/`, gán label theo tên folder (10 loài). Kết hợp `Resize`, `ToTensor`, `DataLoader` — mô phỏng `ImageFolder` nhưng **tự code** để hiểu pipeline. |
| **`learn.py`** | Script ngắn: gọi `ImageFolder("animals/train")`, in `class_to_idx`, mở 1 ảnh bằng PIL — kiểm tra dataset Animals nhanh. |
| **`image/image.py`** | Đọc `image.png` bằng OpenCV (BGR), zero 2 kênh R/G, hiển thị — làm quen **không gian màu BGR vs RGB** trước khi vào PyTorch. |

### Khác

| File | Vai trò |
| :--- | :--- |
| **`.gitignore`** | Bỏ qua `data/`, `animals/`, `cifar10-neural-network-training/data/`, cache Python, model weights — repo GitHub chỉ giữ code. |
| **`README.md`** | Tài liệu repo (file này). |

---

## Luồng học trong repo

```
1. image/image.py          → ảnh là gì (OpenCV)
2. Dataloader_learn.py     → DataLoader + CIFAR10 có sẵn
3. dataset.py              → tự viết Dataset CIFAR
4. models.py               → định nghĩa MLP
5. train_neural_network.py → train + đánh giá (kết quả trên)
6. imageforder_manual.py   → Dataset folder-based (Animals-10)
7. learn.py                  → ImageFolder demo
```

---

## Cài đặt & chạy

```bash
pip install torch torchvision numpy scikit-learn pillow opencv-python
```

**CIFAR-10:** [tải tại đây](https://www.cs.toronto.edu/~kriz/cifar.html) → giải nén:

```
cifar10-neural-network-training/data/cifar-10-batches-py/
```

**Train & xem report:**

```bash
cd cifar10-neural-network-training
python train_neural_network.py
```

**Animals-10:** giải nén vào `animals/train/` + `animals/test/` (cấu trúc folder theo class).

---

_PyTorch · CIFAR-10 · Animals-10 · trained on RTX 4060 Laptop_
