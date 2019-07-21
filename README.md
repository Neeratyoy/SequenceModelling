# DL Lab Project

Final project submission for Deep Learning Lab SS19, University of Freiburg.
Undertaken by Neeratyoy Mallik & [Ashwin Raaghav Narayanan](https://github.com/ashraaghav) under the supervision of [Jorg Frenke](https://github.com/joergfranke).

### Comparison of LSTMs and Transformers in Sequence Modelling

The main objective of this work was to compare the across various sequence modelling tasks, namely
* Sequence Labelling (many-to-one)
    * Toy Task - Learning an arbitrary XOR function over a sequence of 0/1 bit stream
    * Benchmark - Sentiment Analysis on the IMDb dataset
* Sequence to Sequence-same (many-to-many-same)
    * Toy Task - Learning to reverse a sequence of 0/1 bit stream
    * Benchmark - Facebook bAbi task 2
* Sequence to Sequence-different (many-to-many-different)
    * Toy Task - Learning to copy and repeat a sequence of 0/1 bit stream
    * Benchmark - Learning to copy and repeat a sequence of 0/1 bit stream an arbitrary n-times (n > length of training input)

We implemented [LSTM](https://github.com/Neeratyoy/SequenceModelling/blob/master/src/lstm.py) and [Transformer](https://github.com/Neeratyoy/SequenceModelling/blob/master/src/transformer.py) modules from scratch using PyTorch tensors.
Our implementations were verified against PyTorch modules using the Toy Tasks. The results of which can be found in notebooks/ with the prefix _[TOY]_.

Subsequently, each of the 3 aforementioned tasks were solved using the LSTM and Transformer.

    
