# Deep Learning for Computer Vision

Repo cá nhân học PyTorch & Computer Vision.

## Cấu trúc

- `dataset.py` — Custom `Dataset` cho CIFAR-10.
- `Dataloader_learn.py` — DataLoader với CIFAR-10 (torchvision).
- `imageforder_manual.py` — Custom `Dataset` cho Animals-10 (ImageFolder thủ công).
- `learn.py` — Code tập luyện / thử nghiệm.
- `image/` — script tạo & ảnh kết quả.
- `data/` _(gitignored)_ — chứa CIFAR-10.
- `animals/` _(gitignored)_ — chứa Animals-10.

## Cài đặt

```bash
pip install torch torchvision numpy pillow opencv-python
```

## Tải dataset CIFAR-10

Tải `cifar-10-python.tar.gz` từ https://www.cs.toronto.edu/~kriz/cifar.html rồi giải nén vào `data/` sao cho có cấu trúc:

```
data/cifar-10-batches-py/
  data_batch_1 ... data_batch_5
  test_batch
  batches.meta
```

## Chạy

```bash
python dataset.py
```