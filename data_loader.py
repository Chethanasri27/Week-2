import os
import numpy as np
from PIL import Image
import pickle
from config import TRAIN_DIR, TEST_DIR, IMG_SIZE, IMG_CHANNELS, SEED

np.random.seed(SEED)

class DataLoader:
    def __init__(self, train_dir=TRAIN_DIR, test_dir=TEST_DIR):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
    
    def load_data(self, data_dir):
        """Load images from directory structure where each subfolder is a class."""
        images = []
        labels = []
        
        for label in sorted(os.listdir(data_dir)):
            label_dir = os.path.join(data_dir, label)
            if not os.path.isdir(label_dir):
                continue
            
            for img_name in os.listdir(label_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg', '.ppm')):
                    img_path = os.path.join(label_dir, img_name)
                    try:
                        img = Image.open(img_path)
                        img = img.resize((IMG_SIZE, IMG_SIZE))
                        img_array = np.array(img)
                        images.append(img_array)
                        labels.append(int(label))
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
        
        return np.array(images), np.array(labels)
    
    def preprocess_images(self, images):
        """Normalize images to range [0, 1]."""
        return images.astype('float32') / 255.0
    
    def load_and_preprocess(self):
        """Load and preprocess both training and test data."""
        print("Loading training data...")
        self.X_train, self.y_train = self.load_data(self.train_dir)
        self.X_train = self.preprocess_images(self.X_train)
        print(f"Training data shape: {self.X_train.shape}, Labels: {self.y_train.shape}")
        
        print("Loading test data...")
        self.X_test, self.y_test = self.load_data(self.test_dir)
        self.X_test = self.preprocess_images(self.X_test)
        print(f"Test data shape: {self.X_test.shape}, Labels: {self.y_test.shape}")
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def save_data(self, filename):
        """Save preprocessed data to pickle file."""
        data = {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_test': self.X_test,
            'y_test': self.y_test
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {filename}")
    
    def load_data_from_pickle(self, filename):
        """Load preprocessed data from pickle file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        print(f"Data loaded from {filename}")
        return self.X_train, self.y_train, self.X_test, self.y_test
