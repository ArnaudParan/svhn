#!/bin/bash

curl "http://ufldl.stanford.edu/housenumbers/train.tar.gz" | tar xf -C data/svhn/ -&
curl "http://ufldl.stanford.edu/housenumbers/test.tar.gz" | tar xf -C data/svhn/ -&
curl "http://ufldl.stanford.edu/housenumbers/extra.tar.gz" | tar xf -C data/svhn/ -&
curl "http://ufldl.stanford.edu/housenumbers/train_32x32.mat" > data/svhn/train_32x32.mat
curl "http://ufldl.stanford.edu/housenumbers/test_32x32.mat" > data/svhn/test_32x32.mat
curl "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat" > data/svhn/extra_32x32.mat
