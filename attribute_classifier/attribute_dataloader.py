from torch.utils.data import DataLoader, Dataset


def get_attribute_dataloader(datapath,
                             tokenizer,
                             max_length: int = 256,
                             batch_size: int = 32,
                             is_train: bool = False,
                             num_workers: int = None):
    dataset = AttributeDataset(datapath, tokenizer, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
    )


class AttributeDataset(Dataset):

    def __init__(self, datapath: str, tokenizer, max_length: int = 256) -> None:
        data = load(datapath)  # TODO

        self.labels = data['labels']
        self.texts = tokenizer(data['texts'],
                               return_tensors=True,
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