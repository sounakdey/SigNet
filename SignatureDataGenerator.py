import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import image
from scipy import linalg
import warnings
from keras import backend as K
import getpass as gp
import random
random.seed(1337)

class SignatureDataGenerator(object):    
    
    def __init__(self, dataset, tot_writers, num_train_writers, num_valid_writers,
                 num_test_writers, nsamples, batch_sz, img_height, img_width,
                 featurewise_center=False,
                 featurewise_std_normalization=False,
                 zca_whitening=False):
        
        # check whether the total number of writers are less than num_train_writers + num_valid_writers + num_test_writers
        assert tot_writers >= num_train_writers + num_valid_writers + num_test_writers, 'Total writers is less than train and test writers'
        
        self.featurewise_center = featurewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.zca_whitening = zca_whitening
        
        usr = gp.getuser()        
        
        if(dataset=='GPDS960' or dataset=='GPDS300' or dataset == 'Hindi' or dataset == 'Bengali'):
            size = 996            
        elif(dataset=='CEDAR1'):
            size = 852
                   
#        self.image_dir = '/home/' + usr + '/Workspace/SignatureVerification/Datasets/' + dataset + '/'
        self.image_dir = '/home/' + usr + '/Workspace/Datasets/' + dataset + '/'
        data_file = self.image_dir + dataset + '_pairs.txt'
        
        idx_writers = list(range(tot_writers))
        
        idx_train_writers = sorted(np.random.choice(idx_writers, num_train_writers, replace=False))
        idx_valid_writers = sorted(np.random.choice([x for x in idx_writers if x not in idx_train_writers], num_valid_writers, replace=False))
        idx_test_writers = sorted(np.random.choice([x for x in idx_writers if x not in idx_train_writers and x not in idx_valid_writers], num_test_writers, replace=False))
        
        idx_train_lines = []
        for iw in idx_train_writers:
            idx_train_lines += list(range(iw * size, (iw + 1) * size))
            
        idx_valid_lines = []
        for iw in idx_valid_writers:
            idx_valid_lines += list(range(iw * size, (iw + 1) * size))
        
        idx_test_lines = []
        for iw in idx_test_writers:
            idx_test_lines += list(range(iw * size, (iw + 1) * size))
            
        f = open( data_file, 'r' )
        lines = f.readlines()
        f.close()        

        
        # with open('/home/sounak/Dropbox/signature_verification/gpds_pairs_icdar_train.txt') as f:
        #     train_lines = f.read().splitlines()


        train_lines = [lines[i] for i in idx_train_lines]
        valid_lines = [lines[i] for i in idx_valid_lines]
        # test_lines = [lines[i] for i in idx_test_lines]
        # small hack for stupid comment from ICDAR2017
        with open('/home/sounak/Dropbox/signature_verification/gpds_pairs_icdar_test.txt') as f:
            test_lines = f.read().splitlines()

        del lines
        
        # for train writers        
        self.train_lines = self.arrange_lines(train_lines, nsamples, size)
        # self.train_lines = self.arrange_lines(train_lines, nsamples, 552) 
        # for valid writers        
        self.valid_lines = self.arrange_lines(valid_lines, nsamples, size)
        # for test writers        
        # self.test_lines = self.arrange_lines(test_lines, nsamples, size)
        self.test_lines = self.arrange_lines(test_lines, nsamples, 552)  
        
        # Set other parameters
        self.height=img_height
        self.width=img_width
        self.input_shape=(self.height, self.width, 1)
        self.cur_train_index = 0
        self.cur_valid_index = 0
        self.cur_test_index = 0
        self.batch_sz = batch_sz
        self.samples_per_train = 2*nsamples*num_train_writers
        self.samples_per_valid = 2*nsamples*num_valid_writers
        self.samples_per_test = 2*nsamples*num_test_writers
        # Incase dim_ordering = 'tf'
        self.channel_axis = 3
        self.row_axis = 1
        self.col_axis = 2
        
        self.train_labels = np.array([float(line.split(' ')[2].strip('\n')) for line in self.train_lines])
        self.valid_labels = np.array([float(line.split(' ')[2].strip('\n')) for line in self.valid_lines])
        self.test_labels = np.array([float(line.split(' ')[2].strip('\n')) for line in self.test_lines])
        
    def arrange_lines(self, lines, nsamples, size):
        
        idx_lines = []
    
        lp = []
        lin = []
    
        for iline, line in enumerate(lines):
            
            file1, file2, label = line.split(' ')
            
            label = int(label)
            
            lp += [label]        
            lin += [iline]
            
            if(len(lp) != 0 and len(lp) % size == 0):                
                            
                idx1 = [i for i, x in enumerate(lp) if x == 1]
                idx2 = [i for i, x in enumerate(lp) if x == 0]
                
                idx1 = np.random.choice(idx1, nsamples)
                idx2 = np.random.choice(idx2, nsamples)
                
                idx = [None]*(len(idx1)+len(idx2))
                
                idx[::2] = idx1
                idx[1::2] = idx2
                
                del idx1
                del idx2
                
                idx_lines += [lin[i] for i in idx]
                
                lp = []
                lin = []            
            
        lines = [lines[i] for i in idx_lines]

        just_1 = lines[0:][::2]
        just_0 = lines[1:][::2]
        random.shuffle(just_1)
        random.shuffle(just_0)
        lines= [item for sublist in zip(just_1,just_0) for item in sublist]
        
        return lines
        
    def next_train(self):
        while True:
            
            if self.cur_train_index == self.samples_per_train:
                self.cur_train_index = 0
                
            cur_train_index = self.cur_train_index + self.batch_sz
            
            if cur_train_index > self.samples_per_train:
                cur_train_index = self.samples_per_train

            # print 'Cur Index', self.cur_train_index,'\n'
            # print 'End Index', cur_train_index, '\n'
            idx = list(range(self.cur_train_index, cur_train_index))
            # print idx, '\n'
            # print len(self.train_lines)
            
            lines = [self.train_lines[i] for i in idx]
            
            image_pairs = []
            label_pairs = []
                            
            for line in lines:
                file1, file2, label = line.split(' ')
                
                img1 = image.load_img(self.image_dir + file1, grayscale = True,
                target_size=(self.height, self.width))
                                
                img1 = image.img_to_array(img1)#, dim_ordering='tf')
                
                img1 = self.standardize(img1)
                                
                img2 = image.load_img(self.image_dir + file2, grayscale = True,
                target_size=(self.height, self.width))
                
                img2 = image.img_to_array(img2)#, dim_ordering='tf')
                
                img2 = self.standardize(img2)
                
                image_pairs += [[img1, img2]]
                label_pairs += [int(label)]
                
            self.cur_train_index = cur_train_index
            
            images = [np.array(image_pairs)[:,0], np.array(image_pairs)[:,1]]
            labels = np.array(label_pairs)
            yield(images,labels)
            
    def next_valid(self):
        while True:            
            
            if self.cur_valid_index == self.samples_per_valid:
                self.cur_valid_index = 0
                
            cur_valid_index = self.cur_valid_index + self.batch_sz
            
            if cur_valid_index > self.samples_per_valid:
                cur_valid_index = self.samples_per_valid            
            
            idx = list(range(self.cur_valid_index, cur_valid_index))
            
            lines = [self.valid_lines[i] for i in idx]
            
            image_pairs = []
            label_pairs = []
                
            for line in lines:
                file1, file2, label = line.split(' ')
                
                img1 = image.load_img(self.image_dir + file1, grayscale = True,
                target_size=(self.height, self.width))
                
                img1 = image.img_to_array(img1)#, dim_ordering='tf')
                
                img1 = self.standardize(img1)
                                
                img2 = image.load_img(self.image_dir + file2, grayscale = True,
                target_size=(self.height, self.width))
                
                img2 = image.img_to_array(img2)#, dim_ordering='tf')
                
                img2 = self.standardize(img2)
                
                image_pairs += [[img1, img2]]
                label_pairs += [int(label)]

            self.cur_valid_index = cur_valid_index
            
            images = [np.array(image_pairs)[:,0], np.array(image_pairs)[:,1]]
            labels = np.array(label_pairs)
            yield(images,labels)
                
    def next_test(self):
        while True:
            
            if self.cur_test_index == self.samples_per_test:
                self.cur_test_index = 0
                
            cur_test_index = self.cur_test_index + self.batch_sz
            
            if cur_test_index > self.samples_per_test:
                cur_test_index = self.samples_per_test
                            
            idx = list(range(self.cur_test_index, cur_test_index))
            
            lines = [self.test_lines[i] for i in idx]
            
            image_pairs = []
                
            for line in lines:
                file1, file2, label = line.split(' ')
                
                img1 = image.load_img(self.image_dir + file1, grayscale = True,
                target_size=(self.height, self.width))
                
                img1 = image.img_to_array(img1)#, dim_ordering='tf')
                
                img1 = self.standardize(img1)
                
                img2 = image.load_img(self.image_dir + file2, grayscale = True,
                target_size=(self.height, self.width))
                
                img2 = image.img_to_array(img2)#, dim_ordering='tf')
                
                img2 = self.standardize(img2)
                
                image_pairs += [[img1, img2]]

            self.cur_test_index = cur_test_index
            
            images = [np.array(image_pairs)[:,0], np.array(image_pairs)[:,1]]
            yield(images)
            
    def fit(self, x, augment=False, rounds=1, seed=None):
        
        x = np.asarray(x, dtype=K.floatx())
        if x.ndim != 4:
            raise ValueError('Input to `.fit()` should have rank 4. '
                             'Got array with shape: ' + str(x.shape))
        if x.shape[self.channel_axis] not in {1, 3, 4}:
            raise ValueError(
                'Expected input to be images (as Numpy array) '
                'following the dimension ordering convention "' + self.dim_ordering + '" '
                '(channels on axis ' + str(self.channel_axis) + '), i.e. expected '
                'either 1, 3 or 4 channels on axis ' + str(self.channel_axis) + '. '
                'However, it was passed an array with shape ' + str(x.shape) +
                ' (' + str(x.shape[self.channel_axis]) + ' channels).')

        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if augment:
            ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]), dtype=K.floatx())
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= (self.std + K.epsilon())

        if self.zca_whitening:
            flat_x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + 10e-7))), u.T)
            
    def standardize(self, x):
        
        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-7)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (x.size))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        return x
