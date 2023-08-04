import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class DataGenerator:
    def __init__(self, filenames, labels, target_size=(128, 128), batch=32):
        self.filenames = filenames
        self.labels = labels
        self.batch = batch
        self.target_size = target_size

        # Batch lists
        self.batch_filenames = []
        self.batch_labels = []

        # Calculate the number of epochs
        self.epochs = len(filenames) // self.batch
        self.epochs += 0 if (len(self.filenames) % self.batch) == 0 else 1

        # Iterator attributes
        self.current_epoch = 0
        self.get_batches()

    def __next__(self):
        if self.current_epoch >= self.epochs:
            self.current_epoch = 0
            self.get_batches()
            raise StopIteration

        images = []
        labels = self.batch_labels[self.current_epoch]

        for filename in self.batch_filenames[self.current_epoch]:
            img_arr = img_to_array(load_img(filename, target_size=self.target_size))
            images.append(img_arr)

        self.current_epoch += 1
        return np.array(images, dtype=np.int32), labels

    def __iter__(self):
        self.current_epoch = 0
        self.get_batches()
        return self

    def get_batches(self):
        random_index = np.arange(len(self.filenames))
        np.random.shuffle(random_index)
        shuffled_filenames = self.filenames[random_index]
        shuffled_labels = self.labels[random_index]

        self.batch_filenames = []
        self.batch_labels = []

        for i in range(self.epochs):
            self.batch_filenames.append(shuffled_filenames[i:i + self.batch])
            self.batch_labels.append(shuffled_labels[i:i + self.batch])

        remainder = len(self.filenames) % self.batch
        if remainder != 0:
            self.batch_filenames.append(shuffled_filenames[-remainder:])
            self.batch_labels.append(shuffled_labels[-remainder:])
