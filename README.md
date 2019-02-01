# pytorch-NetVlad

Implementation of [NetVlad](https://arxiv.org/abs/1511.07247) in PyTorch, including code for training the model on the Pittsburgh dataset.

### Reproducing the paper

Below are the result as compared to the results in third row in the right column of Table 1:

|   |R@1|R@5|R@10|
|---|---|---|---|
| [NetVlad paper](https://arxiv.org/abs/1511.07247)  | 84.1  | 94.6  | 95.5  |
| pytorch-NetVlad(alexnet)  | 68.6  | 84.6  | 89.3  |
| pytorch-NetVlad(vgg16)  | 85.2  | 94.8  | 97.0  |

Running main.py with train mode and default settings should give similar scores to the ones shown above. Additionally, the model state for the above run is
available here: https://drive.google.com/open?id=17luTjZFCX639guSVy00OUtzfTQo4AMF2

Using this checkpoint and the following command you can obtain the results shown above:

    python main.py --mode=test --split=val --resume=vgg16_netvlad_checkpoint/

# Setup

## Dependencies

1. [PyTorch](https://pytorch.org/get-started/locally/) (at least v0.4.0)
2. [Faiss](https://github.com/facebookresearch/faiss)
3. [scipy](https://www.scipy.org/)
    - [numpy](http://www.numpy.org/)
    - [sklearn](https://scikit-learn.org/stable/)
    - [h5py](https://www.h5py.org/)
4. [tensorboardX](https://github.com/lanpa/tensorboardX)

## Data

Running this code requires a copy of the Pittsburgh 250k (available [here](http://www.ok.ctrl.titech.ac.jp/~torii/project/repttile/)), 
and the dataset specifications for the Pittsburgh dataset (available [here](https://www.di.ens.fr/willow/research/netvlad/data/netvlad_v100_datasets.tar.gz)).
`pittsburgh.py` contains a hardcoded path to a directory, where the code expects directories `000` to `010` with the various Pittsburth database images, a directory
`queries_real` with subdirectories `000` to `010` with the query images, and a directory `datasets` with the dataset specifications (.mat files).


# Usage

`main.py` contains the majority of the code, and has three different modes (`train`, `test`, `cluster`) which we'll discuss in mode detail below.

## Train

In order to initialise the NetVlad layer it is necessary to first run `main.py` with the correct settings and `--mode=cluster`. After which a model can be trained using (the following default flags):

    python main.py --mode=train --arch=vgg16 --pooling=netvlad --num_clusters=64

The commandline args, the tensorboard data, and the model state will all be saved to `opt.runsPath`, which subsequently can be used for testing, or to resuming training.

For more information on all commandline arguments run:

    python main.py --help

## Test

To test a previously trained model on the Pittsburgh 30k testset (replace directory with correct dir for your case):

    python main.py --mode=test --resume=runsPath/Nov19_12-00-00_vgg16_netvlad --split=test

The commandline arguments for training were saved, so we shouldnt need to specify them for testing.
Additionally, to obtain the 'off the shelf' performance we can also omit the resume directory:

    python main.py --mode=test

## Cluster

In order to initialise the NetVlad layer we need to first sample from the data and obtain `opt.num_clusters` centroids. This step is
necessary for each configuration of the network and for each dataset. To cluster simply run

    python main.py --mode=cluster --arch=vgg16 --pooling=netvlad --num_clusters=64

with the correct values for any additional commandline arguments.
