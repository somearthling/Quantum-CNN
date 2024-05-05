'''
Has functions to load and pad datasets for quantum embedding
'''

from torchvision import datasets
import numpy as np
from torch.utils import data
from jax.tree_util import tree_map


def numpy_collate(batch):
    return tree_map(np.asarray, data.default_collate(batch))
    
class PreprocessAndPad(object):
    '''
    Pads images to a power of 2 in each dimension and repeats them for the number of filters
    '''

    def __init__(self, filters=1):
        self.filters = filters

    def __call__(self, img):
        # Convert the image to a numpy array
        img = np.array(img)

        # Pad the image to the nearest power of 2
        shape = np.array(img.shape)
        pad_shape = 2**np.ceil(np.log2(shape)).astype(int)
        total_padding = pad_shape - shape
        padding = np.stack([total_padding // 2, -(total_padding // -2)], axis=1)
        padded_img = np.pad(img, padding, "constant", constant_values=0)

        # Repeat the image for the number of filters
        processed_img = np.repeat(padded_img, self.filters, axis=1)

        # Unroll the image into a 1D array
        processed_img = processed_img.flatten()

        return processed_img.astype(float)

class FilteredDataset(data.Dataset):
    '''
    A dataset that only returns items with labels in the specified classes
    '''
    def __init__(self, dataset, classes):
        self.dataset = dataset
        self.classes = set(range(classes))
        self.indices = [i for i, (_, label) in enumerate(dataset) if label in self.classes]

    def __getitem__(self, index):
        # Use the precomputed indices to access the items
        image, label = self.dataset[self.indices[index]]
        return image, label

    def __len__(self):
        # The length of the dataset is now the number of matching items
        return len(self.indices)
    

class RepeatTruncateSampler(data.Sampler):
    def __init__(self, data_source, num_samples):
        self.num_samples = num_samples
        self.data_source = data_source
        super().__init__(data_source)

    def __iter__(self):
        return iter((list(range(len(self.data_source))) * self.num_samples)[:self.num_samples])

    def __len__(self):
        return self.num_samples


class LoadData(data.DataLoader):
    '''
    Loads and pads datasets for quantum embedding, converting their images to numpy arrays.
    If the number of classes k is specified, only the first k classes will be used.
    If the number of iterations is specified, the dataloader will only return that many batches by repitition or truncation.
    '''
    def __init__(self, dataset, batch_size=1,
               train=True, filters=1, 
               classes=None, iters=None,
               shuffle=False, sampler=None,
               batch_sampler=None, num_workers=0,
               pin_memory=False, drop_last=True,
               timeout=0, worker_init_fn=None):
        try:
            dataset_function = getattr(datasets, dataset)(root="data", train=train, download=True, transform=PreprocessAndPad(filters=filters))
        except AttributeError as exc:
            raise ValueError(f"Dataset {dataset} not found in torchvision.datasets") from exc
        except TypeError:
            if dataset == "EMNIST":
                dataset_function = datasets.EMNIST(root="data", split="digits", train=train, download=True, transform=PreprocessAndPad(filters=filters))

        if classes is not None:
            dataset_function = FilteredDataset(dataset_function, classes)

        if iters is not None and sampler is not None:
            raise ValueError("Cannot specify both iters and sampler")

        if iters is not None:
            sampler = RepeatTruncateSampler(dataset_function, iters * batch_size)

        self.dataset_name = dataset
        self.filters = filters
        self.image_shape = dataset_function[0][0].shape
        self.datapoints = len(dataset_function)
        self.classes = classes or len(dataset_function.classes)

        super().__init__(dataset_function,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         sampler=sampler,
                         batch_sampler=batch_sampler,
                         num_workers=num_workers,
                         collate_fn=numpy_collate,
                         pin_memory=pin_memory,
                         drop_last=drop_last,
                         timeout=timeout,
                         worker_init_fn=worker_init_fn)