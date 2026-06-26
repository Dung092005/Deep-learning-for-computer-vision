from torch.utils.data import Dataset
import os
import pickle
import numpy as np

class MyDataset(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        if train == True:
            data_files = []
            for i in range(1, 6):
                path = os.path.join(root, "data_batch_{}".format(i))
                data_files.append(path)
        else:
            data_files = [os.path.join(root, "test_batch")]

        self.images = []
        self.labels = []
        for data_file in data_files:
            with open(data_file, "rb") as fo:
                data = pickle.load(fo, encoding="bytes")
                self.images.extend(data[b"data"])
                self.labels.extend(data[b"labels"])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = self.images[item].reshape((3, 32, 32)).astype(np.float32)  # RGB
        label = self.labels[item]
        return image / 255., label

if __name__ == "__main__":
    from pathlib import Path

    data_dir = Path(__file__).resolve().parent / "data" / "cifar-10-batches-py"
    dataset = MyDataset(root=str(data_dir), train=True)
    image, label = dataset[234]
    print(image.shape)
    print(label)