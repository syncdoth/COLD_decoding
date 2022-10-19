from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


def get_attribute_dataloader(dataname,
                             tokenizer,
                             max_length: int = 256,
                             batch_size: int = 32,
                             split: str = 'test',
                             num_workers: int = None):
    dataset = AttributeDataset(dataname, tokenizer, max_length=max_length, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == 'train',
        num_workers=num_workers,
    )


class AttributeDataset(Dataset):

    def __init__(self,
                 dataname: str,
                 tokenizer,
                 max_length: int = 256,
                 split: str = 'test') -> None:
        data = load_dataset(dataname)
        data = data[split]

        self.labels = data['label']
        self.texts = tokenizer(data['sentence'],
                               return_tensors='pt',
                               padding='max_length',
                               truncation=True,
                               max_length=max_length)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.texts[idx]
        label = self.labels[idx]

        data['labels'] = label

        return data