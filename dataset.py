from torch.utils.data import Dataset
import os
import pickle


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
        image = self.images[item]
        label = self.labels[item]
        return image, label


if __name__ == "__main__":
    dataset = MyDataset(root="data/cifar-10-batch", train=True)
    image, label = dataset[234]
    print(image.shape)
    print(label)