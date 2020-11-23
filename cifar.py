import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import color

def _unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d

def _reshape_images(images):
    return images.reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1])
    
def normalize(images):
    return np.divide(images, np.float32(255))

def avg_color(images):
    return np.mean(images, axis = (1,2), dtype = np.float32)

def color_hist(image, num_bins):
    image = image.reshape([-1, 3])
    hist, _ = np.histogramdd(image, range = [(0,1), (0,1), (0,1)], bins = num_bins)
    hist = hist / (32 ** 2)
    hist = np.reshape(hist, -1)
    return hist

def edges(image):
    img = color.rgb2gray(image)
    img = np.multiply(img, np.float32(255)).astype(np.uint8)
    edges = cv2.Canny(img, 100, 200, L2gradient=True)
    plt.imshow(edges, cmap = 'Greys')
    plt.show()
    


def distance(images1, images2):
    from scipy.spatial.distance import cdist

    return cdist(images1, images2)

def load_training_data(path_to_dataset):
    import os
    
    training_data = {"images": [], "labels": [], "filenames": []}
    
    for i in range(1,6):
        batch = _unpickle(
            os.path.join(path_to_dataset, "data_batch_{}".format(i)))
        
        training_data["images"].append(batch[b"data"])
        training_data["labels"].extend(batch[b"labels"])
        training_data["filenames"].extend(map(bytes.decode, batch[b"filenames"]))
    
    training_data["images"] = np.concatenate(training_data["images"], axis=0)
    training_data["images"] = _reshape_images(training_data["images"])
    return training_data

def load_test_data(path_to_dataset):
    import os
    
    batch = _unpickle(os.path.join(path_to_dataset, "test_batch"))
    
    test_data = {}
    test_data["images"] = batch[b"data"]
    test_data["labels"] = batch[b"labels"]
    test_data["filenames"] = list(map(bytes.decode, batch[b"filenames"]))
    
    test_data["images"] = _reshape_images(test_data["images"])
    return test_data