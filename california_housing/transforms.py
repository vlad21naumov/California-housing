def normalize_data(train_dataset, val_dataset):
    mean = train_dataset.data.mean(axis=0)
    std = train_dataset.data.std(axis=0)
    train_dataset.data = (train_dataset.data - mean) / std
    val_dataset.data = (val_dataset.data - mean) / std
    return train_dataset, train_dataset
