import cv2
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import os
import shutil

IMAGE_SIZE = 28
NUM_CLASSES = 10
THICKNESS_RANGE = (1, 5)
COLOR_RANGE = (1, 256)


def get_data() -> None:
    transform = transforms.Compose([transforms.ToTensor()])

    if os.path.exists('mnist+.csv'):
        print("The data has been downloaded.")
    else:
        trainset = torchvision.datasets.MNIST(root='.', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='.', train=False, download=True, transform=transform)

        full_dataset = trainset + testset

        def convert_to_numpy(dataset):
            images = np.array([np.array(image[0]).flatten() for image in dataset])
            labels = np.array([image[1] for image in dataset])
            return images, labels

        def generate_images(num_images):
            generated = []
            for _ in range(num_images):
                thickness = np.random.randint(1, 5)
                canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
                for _ in range(5 - thickness):
                    x1, y1 = np.random.randint(0, 28, size=2)
                    x2, y2 = np.random.randint(0, 28, size=2)
                    color = np.random.randint(1, 256)
                    cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)
                    canvas = canvas / 255.0
                    generated.append(canvas.flatten())
            return np.array(generated)

        images, labels = convert_to_numpy(full_dataset)
        num_generated_images = len(full_dataset) // (NUM_CLASSES + 1)
        generated_images = generate_images(num_generated_images)

        images = np.concatenate((images, generated_images))
        labels = np.concatenate((labels, [NUM_CLASSES] * num_generated_images))

        df = pd.DataFrame(images)
        df['label'] = labels

        df.to_csv('mnist+.csv', index=False)
        shutil.rmtree('./MNIST')
