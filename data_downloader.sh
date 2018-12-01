#!/bin/bash

curl "http://ufldl.stanford.edu/housenumbers/train.tar.gz" | tar xf -C data/svhn/ -&
curl "http://ufldl.stanford.edu/housenumbers/test.tar.gz" | tar xf -C data/svhn/ -&
curl "http://ufldl.stanford.edu/housenumbers/extra.tar.gz" | tar xf -C data/svhn/ -&
