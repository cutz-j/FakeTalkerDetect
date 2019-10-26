import numpy as np
import os
from PIL import Image

def generator(directory, batch_size=32):
    folder =  np.sort(os.listdir(directory))
    real_img = np.asarray(glob.glob(directory + '/' + folder[0]+'/*.png'))
    real_idx = np.arange(len(real_img))
    
    while 1:
        X1 = []
        X2 = []
        y = []
        
        if (len(real_idx) < batch_size):
            real_idx = np.arange(len(real_img))
            continue
        
        for _ in range(batch_size):
            if (len(real_idx) < batch_size):
                real_idx = np.arange(len(real_img))
                break
            random1 = np.random.choice(real_idx, 1, replace=False)
            real_idx = real_idx[~np.isin(real_idx, random1)]
            random2 = np.random.choice(real_idx, 1, replace=False)
            real_idx = real_idx[~np.isin(real_idx, random2)]
            X1.append(np.asarray(Image.open(real_img[random1[0]]).convert("RGB"))/255.)
            X2.append(np.asarray(Image.open(real_img[random2[0]]).convert("RGB"))/255.)
            y.append(np.array([0.]))

        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        y = np.asarray(y)
        yield [X1, X2], y
        
def generator_res(ft_dir, directory, batch_size=1, critical_value=0.5):
    folder = np.sort(os.listdir(directory))
    ft_img = np.asarray(glob.glob(ft_dir + '/' + '0' +'/*.png'))
    ft_idx = np.arange(len(ft_img))
    random1 = np.random.choice(ft_idx, 1, replace=False)
    img = np.asarray(Image.open(ft_img[random1[0]]).convert("RGB"))/255.
    fake_img = np.asarray(glob.glob(directory + '/' + folder[1] + '/*.png'))
    fake_idx = np.arange(len(fake_img))
    real_img = np.asarray(glob.glob(directory + '/' + folder[0] + '/*.png'))
    real_idx = np.arange(len(real_img))
    while 1:
        X1 = []
        X2 = []
        y = []
        if (len(fake_idx) < batch_size):
            break
        if (len(real_idx) < batch_size):
            break
        for _ in range(batch_size):
            if np.random.uniform() > critical_value:
                if (len(fake_idx) < batch_size):
                    break
                random2 = np.random.choice(fake_idx, 1, replace=False)
                fake_idx = fake_idx[~np.isin(fake_idx, random2)]
                X1.append(img)
                X2.append(np.asarray(Image.open(fake_img[random2[0]]).convert("RGB"))/255.)
                y.append(np.array([1.]))
            else:
                if (len(real_idx) < batch_size):
                    break
                random3 = np.random.choice(real_idx, 1, replace=False)
                real_idx = real_idx[~np.isin(real_idx, random3)]
                X1.append(img)
                X2.append(np.asarray(Image.open(real_img[random3[0]]).convert("RGB"))/255.)
                y.append(np.array([0.]))
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        y = np.asarray(y)
        yield [X1, X2], y