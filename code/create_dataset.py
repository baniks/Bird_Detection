import tensorflow as tf
import numpy as np
import cv2
import glob
from random import shuffle
import sys
import os
from sklearn.model_selection import train_test_split


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_image(addr):
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def create_data_record(out_file_name, addrs, labels):
    
    # open the TFRecords file writer
    writer = tf.python_io.TFRecordWriter(out_file_name)
    
    for i in range(len(addrs)):
        if i%1000==0:
            print('Train data: {}/{}'.format(i,len(addrs)))
            sys.stdout.flush()
        
        img = load_image(addrs[i])
        label = labels[i]
        
        if img is None:
            continue
        
        # create a feature
        feature = {
            'image_raw':_bytes_feature(img.tostring()),
            'label': _int64_feature(label)
        }
        
        # tensorflow structure for storing each data sample
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # serialize the data sample and write to file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()



if __name__ == '__main__':
    project_root = '../'
    data_path = '/informatik2/students/home/4banik/Documents/datasets/CUB_200/images/*/*.jpg'

    addrs = glob.glob(data_path)
    labels = []
    for addr in addrs:
        dir_path = os.path.dirname(addr)
        dir_name = os.path.basename(dir_path)
        file_label = int(dir_name.split('.')[0])
        labels.append(file_label)
    
    # Split train, val, test set ( shuffle = True by default )
    X, X_test, Y, Y_test = train_test_split(addrs, labels, test_size=0.20, random_state=21)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25, random_state=21)

    np.savetxt(project_root+'data/X_train.txt', X_train, fmt="%s")
    np.savetxt(project_root+'data/X_val.txt', X_val, fmt="%s")
    np.savetxt(project_root+'data/X_test.txt', X_test, fmt="%s")
    np.savetxt(project_root+'data/Y_train.txt', Y_train, fmt="%i")
    np.savetxt(project_root+'data/Y_val.txt', Y_val, fmt="%i")
    np.savetxt(project_root+'data/Y_test.txt', Y_test, fmt="%i")
    
    # create tensorflow record of the train/val/test set
    create_data_record('train.tfrecords', X_train, Y_train)
    create_data_record('val.tfrecords', X_val, Y_val)
    create_data_record('test.tfrecords', X_test, Y_test)
