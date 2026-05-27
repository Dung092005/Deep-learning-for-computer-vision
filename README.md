# Deep Learning for Computer Vision

Repo cá nhân học PyTorch & Computer Vision.

## Cấu trúc

- `dataset.py` — Custom `Dataset` cho CIFAR-10.
- `learn.py` — Code tập luyện / thử nghiệm.
- `image/` — script tạo & ảnh kết quả.
- `data/` _(gitignored)_ — chứa CIFAR-10.
- `animals/` _(gitignored)_ — chứa Animals-10.

## Cài đặt

```bash
pip install torch numpy
```

## Tải dataset CIFAR-10

Tải `cifar-10-python.tar.gz` từ https://www.cs.toronto.edu/~kriz/cifar.html rồi giải nén vào `data/` sao cho có cấu trúc:

```
data/cifar-10-batch/
  data_batch_1 ... data_batch_5
  test_batch
```

## Chạy

```bash
python dataset.py
```