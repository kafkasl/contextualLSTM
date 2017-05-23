To compile .cc operation:

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC -I $TF_INC -O2
