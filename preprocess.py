#####################################################
# No need to run this file, if you download dataset #
#####################################################

import os, shutil
import glob
import tensorflow as tf
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# Devide real and fake (train and validation)
base_dir = '/home/skkulab/ICCV/Dataset/preprocessed_dataset'

# train, valid, data directory
train_dir = os.path.join(base_dir, 'train')
shutil.rmtree(train_dir)
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
shutil.rmtree(validation_dir)
os.mkdir(validation_dir)

# train real, fake data
train_real_dir = os.path.join(train_dir, '0_real')
os.mkdir(train_real_dir)
train_fake_dir = os.path.join(train_dir, '1_fake')
os.mkdir(train_fake_dir)

# validation real, fake data
validation_real_dir = os.path.join(validation_dir, '0_real')
os.mkdir(validation_real_dir)
validation_fake_dir = os.path.join(validation_dir, '1_fake')
os.mkdir(validation_fake_dir)

nb_data = 140000

for filename in glob.glob('/home/skkulab/ICCV/Dataset/trainset/*.png'):
    # copy fake data
    if('hat' in filename):
        shutil.copy(filename, train_fake_dir)
    # copy real data
    else:
        shutil.copy(filename, train_real_dir)
        
    nb_real_data = len(next(os.walk(train_real_dir))[2])
    nb_fake_data = len(next(os.walk(train_fake_dir))[2])
    
    # to make real, fake nb same
    if((nb_real_data == nb_data) and (nb_fake_data == nb_data)):
        break        
     
# Split    
print("Real data:", len(next(os.walk(train_real_dir))[2]))
print("Fake data:", len(next(os.walk(train_fake_dir))[2]))

for filename in glob.glob('/home/skkulab/ICCV/Dataset/preprocessed_dataset/train/1_fake/*.png'):
    shutil.move(filename, validation_fake_dir)
    if(len(next(os.walk(validation_fake_dir))[2]) > (nb_fake_data*0.1)):
        break

nb_fake_train_data = len(next(os.walk(train_fake_dir))[2])
nb_fake_validation_data = len(next(os.walk(validation_fake_dir))[2])

print("Fake train data :", nb_fake_train_data)
print("Fake validation data :", nb_fake_validation_data)

base_dir = '/home/skkulab/ICCV/Dataset/preprocessed_dataset'

# test data directory
test_dir = os.path.join(base_dir, 'test')
shutil.rmtree(test_dir)
os.mkdir(test_dir)

# test real, fake data
test_real_dir = os.path.join(test_dir, '0_real')
os.mkdir(test_real_dir)
test_fake_dir = os.path.join(test_dir, '1_fake')
os.mkdir(test_fake_dir)


# devide real, fake data
for filename in glob.glob('/home/skkulab/ICCV/Dataset/testset/*.png'):
    # copy fake data
    if('hat' in filename):
        shutil.copy(filename, test_fake_dir)
    # copy real data
    else:
        shutil.copy(filename, test_real_dir)
        
nb_real_test_data = len(next(os.walk(test_real_dir))[2])
nb_fake_test_data = len(next(os.walk(test_fake_dir))[2])

print("Real test data: ", nb_real_data)
print("Fake test data: ", nb_fake_data)