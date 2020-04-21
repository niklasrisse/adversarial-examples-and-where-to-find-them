import numpy as np
import foolbox
import json
import pickle
from PIL import Image

class L2(foolbox.distances.Distance):

    def _calculate(self):
        min_, max_ = self._bounds
        diff = (self.other - self.reference) / (max_ - min_)
        value = np.linalg.norm(diff.flatten(), ord=2).astype(np.float64)
        gradient = None
        return value, gradient

    @property
    def gradient(self):
        raise NotImplementedError

    def __str__(self):
        return "L2 distance = {}".format(self._value)
    
class L1(foolbox.distances.Distance):

    def _calculate(self):
        min_, max_ = self._bounds
        diff = (self.other - self.reference) / (max_ - min_)
        value = np.linalg.norm(diff.flatten(), ord=1).astype(np.float64)
        gradient = None
        return value, gradient

    @property
    def gradient(self):
        raise NotImplementedError

    def __str__(self):
        return "L1 distance = {}".format(self._value)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

def normalize_per_feature_0_1(X_train, X_test):
    """
    This method is copied from https://github.com/max-andr/provably-robust-boosting/blob/master/data.py
    You can find the LICENSE with copyright notive below this method.
    """
    X_train_max = X_train.max(axis=0, keepdims=True)
    X_train_min = X_train.min(axis=0, keepdims=True)
    X_train = (X_train - X_train_min) / (X_train_max - X_train_min)
    X_test = (X_test - X_train_min) / (X_train_max - X_train_min)
    return X_train, X_test

"""
Copyright (c) 2019, Maksym Andriushchenko and Matthias Hein
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

def har():
    """
    This method is copied from https://github.com/max-andr/provably-robust-boosting/blob/master/data.py
    You can find the LICENSE with copyright notive below this method.
    """
    eps_dataset = 0.025
    path_train, path_test = 'har/train/', 'har/test/'
    X_train, X_test = np.loadtxt(path_train + 'X_train.txt'), np.loadtxt(path_test + 'X_test.txt')
    y_train, y_test = np.loadtxt(path_train + 'y_train.txt'), np.loadtxt(path_test + 'y_test.txt')
    y_train, y_test = y_train - 1, y_test - 1  # make the class numeration start from 0
    X_train, X_test = (X_train + 1) / 2, (X_test + 1) / 2  # from [-1, 1] to [0, 1]
    return X_train, y_train, X_test, y_test, eps_dataset

"""
Copyright (c) 2019, Maksym Andriushchenko and Matthias Hein
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

def tinyimagenet():

    """
    This method is adapted from https://github.com/rmccorm4/Tiny-Imagenet-200/blob/master/networks/data_utils.py
    You can find the LICENSE with copyright notive below this method.
    """
    
    path, wnids_path = 'tiny-imagenet-200', 'tiny-imagenet-200'
    resize='False'
    num_classes=200
    dtype=np.float32
    
    # First load wnids
    wnids_file = os.path.join('wnids' + '.txt')
    with open(os.path.join(path, wnids_file), 'r') as f:
        wnids = [x.strip() for x in f]

    # Map wnids to integer labels
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # Use words.txt to get names for each class
    words_file = os.path.join('words' + '.txt')
    with open(os.path.join(path, words_file), 'r') as f:
        wnid_to_words = dict(line.split('\t') for line in f)
        for wnid, words in wnid_to_words.items():
            wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
    class_names = [wnid_to_words[wnid] for wnid in wnids]

    # Next load training data.
    X_train = []
    y_train = []
    for i, wnid in enumerate(wnids[:num_classes]):
        #if (i + 1) % 20 == 0:
            # print('loading training data for synset %d / %d' % (i + 1, len(wnids)))
        # To figure out the filenames we need to open the boxes file
        boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
        with open(boxes_file, 'r') as f:
            filenames = [x.split('\t')[0] for x in f]
        num_images = len(filenames)

        if resize.lower() == 'true':
            X_train_block = np.zeros((num_images, 3, 32, 32), dtype=dtype)
        else:
            X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)

        y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)
        for j, img_file in enumerate(filenames):
            img_file = os.path.join(path, 'train', wnid, 'images', img_file)
            
            image = Image.open(img_file)
            image = np.asarray(image)
            
            if (image.shape == (64,64,3)):
                image = image.reshape(3,64,64)

            X_train_block[j] = image
            
        X_train.append(X_train_block)
        y_train.append(y_train_block)

    # We need to concatenate all training data
    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.concatenate(y_train, axis=0)

    X_train = X_train/255
    
    cleaned_x_test = []
    cleaned_y_test = []


    for i, point in enumerate(X_train):
        if not math.isnan(np.max(point)):
            cleaned_x_test.append(point)
            cleaned_y_test.append(Y_train[i])
        
    cleaned_x_test = np.array(cleaned_x_test)
    cleaned_y_test = np.array(cleaned_y_test)

    X_train = cleaned_x_test
    Y_train = cleaned_y_test
  
    return X_train, Y_train

"""
The MIT License (MIT)

Copyright (c) 2018 Ryan McCormick

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""