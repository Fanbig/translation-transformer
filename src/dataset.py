# 1.定义Dataset
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import config


class TranslationDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tensor = torch.tensor(self.data[index]['zh'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]['en'], dtype=torch.long)
        return input_tensor, target_tensor


# 2. 提供一个获取dataloader的方法
def collate_fn(batch):
    # batch：二元组列表:[(input_tensor, target_tensor)]
    input_tensors = [item[0] for item in batch]
    target_tensors = [item[1] for item in batch]

    input_tensor = pad_sequence(input_tensors, batch_first=True, padding_value=0)
    target_tensor = pad_sequence(target_tensors, batch_first=True, padding_value=0)

    return input_tensor, target_tensor


def get_dataloader(train=True):
    path = config.PROCESSED_DATA_DIR / ('train.jsonl' if train else 'test.jsonl')
    dataset = TranslationDataset(path)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


if __name__ == '__main__':
    train_dataloader = get_dataloader()
    test_dataloader = get_dataloader(train=False)
    print(len(train_dataloader))
    print(len(test_dataloader))

    for input_tensor, target_tensor in train_dataloader:
        print(input_tensor)  # input_tensor.shape: [batch_size, seq_len]
        print(target_tensor)  # target_tensor.shape : [batch_size]
        break
