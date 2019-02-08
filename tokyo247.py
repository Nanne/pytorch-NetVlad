import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from os.path import join, exists
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image

from sklearn.neighbors import NearestNeighbors
import h5py

root_dir = '/nfs/ibrahimi/data/pittsburgh/'
if not exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to Pittsburgh dataset')

struct_dir = join(root_dir, 'datasets/')
#queries_dir = join(root_dir, 'queries_real')

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

def get_whole_training_set(onlyDB=False):
    structFile = join(struct_dir, 'tokyoTM_train.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform(),
                             onlyDB=onlyDB)

def get_whole_val_set():
    structFile = join(struct_dir, 'tokyoTM_val.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform())

def get_training_query_set(margin=0.1):
    structFile = join(struct_dir, 'tokyoTM_train.mat')
    return QueryDatasetFromStruct(structFile,
                             input_transform=input_transform(), margin=margin)

def get_val_query_set():
    structFile = join(struct_dir, 'tokyoTM_val.mat')
    return QueryDatasetFromStruct(structFile,
                             input_transform=input_transform())

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 
    'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

def parse_dbStruct(path):
    mat = loadmat(path)
    matStruct = mat['dbStruct'].item()

    whichSet = matStruct[0].item()

    dbImage = [f[0].item() for f in matStruct[1]]
    utmDb = matStruct[2].T

    qImage = [f[0].item() for f in matStruct[4]]
    utmQ = matStruct[5].T

    numDb = matStruct[7].item()
    numQ = matStruct[8].item()

    posDistThr = matStruct[9].item()
    posDistSqThr = matStruct[10].item()
    nonTrivPosDistSqThr = matStruct[11].item()

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, 
            utmQ, numDb, numQ, posDistThr, 
            posDistSqThr, nonTrivPosDistSqThr)

class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, input_transform=None, onlyDB=False):
        super().__init__()

        self.input_transform = input_transform

        self.dbStruct = parse_dbStruct(structFile)
        self.images = [join(root_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        if not onlyDB:
            self.images += [join(queries_dir, qIm) for qIm in self.dbStruct.qImage]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None

    def __getitem__(self, index):
        img = Image.open(self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        #fit NN to find them, search by radius
        if  self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.dbStruct.utmDb)

            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ,
                    radius=self.dbStruct.posDistThr)

        return self.positives
        
def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).
    
    Args:
        data: list of tuple (query, positive, negatives). 
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter (lambda x:x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    import itertools
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negCounts, indices

class QueryDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, nNegSample=1000, nNeg=10, margin=0.1, input_transform=None):
        super().__init__()

        self.input_transform = input_transform
        self.margin = margin

        self.dbStruct = parse_dbStruct(structFile)
        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        self.nNegSample = nNegSample # number of negatives to randomly sample
        self.nNeg = nNeg # number of negatives used for training

        # potential positives are those within nontrivial threshold range
        #fit NN to find them, search by radius
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.dbStruct.utmDb)

        # TODO use sqeuclidean as metric?
        self.nontrivial_positives = list(knn.radius_neighbors(self.dbStruct.utmQ,
                radius=self.dbStruct.nonTrivPosDistSqThr**0.5, 
                return_distance=False))
        # radius returns unsorted, sort once now so we dont have to later
        for i,posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)
        # its possible some queries don't have any non trivial potential positives
        # lets filter those out
        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives])>0)[0]

        # potential negatives are those outside of posDistThr range
        potential_positives = knn.radius_neighbors(self.dbStruct.utmQ,
                radius=self.dbStruct.posDistThr, 
                return_distance=False)

        self.potential_negatives = []
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.dbStruct.numDb),
                pos, assume_unique=True))

        self.cache = None # filepath of HDF5 containing feature vectors for images

        self.negCache = [np.empty((0,)) for _ in range(self.dbStruct.numQ)]

    def __getitem__(self, index):
        index = self.queries[index] # re-map index to match dataset
        with h5py.File(self.cache, mode='r') as h5: 
            h5feat = h5.get("features")

            qOffset = self.dbStruct.numDb 
            qFeat = h5feat[index+qOffset]

            posFeat = h5feat[self.nontrivial_positives[index].tolist()]
            knn = NearestNeighbors(n_jobs=-1) # TODO replace with faiss?
            knn.fit(posFeat)
            dPos, posNN = knn.kneighbors(qFeat.reshape(1,-1), 1)
            dPos = dPos.item()
            posIndex = self.nontrivial_positives[index][posNN[0]].item()

            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
            negSample = np.unique(np.concatenate([self.negCache[index], negSample]))

            negFeat = h5feat[negSample.tolist()]
            knn.fit(negFeat)

            dNeg, negNN = knn.kneighbors(qFeat.reshape(1,-1), 
                    self.nNeg*10) # to quote netvlad paper code: 10x is hacky but fine
            dNeg = dNeg.reshape(-1)
            negNN = negNN.reshape(-1)

            # try to find negatives that are within margin, if there aren't any return none
            violatingNeg = dNeg < dPos + self.margin**0.5
     
            if np.sum(violatingNeg) < 1:
                #if none are violating then skip this query
                return None

            negNN = negNN[violatingNeg][:self.nNeg]
            negIndices = negSample[negNN].astype(np.int32)
            self.negCache[index] = negIndices

        query = Image.open(join(queries_dir, self.dbStruct.qImage[index]))
        positive = Image.open(join(root_dir, self.dbStruct.dbImage[posIndex]))

        if self.input_transform:
            query = self.input_transform(query)
            positive = self.input_transform(positive)

        negatives = []
        for negIndex in negIndices:
            negative = Image.open(join(root_dir, self.dbStruct.dbImage[negIndex]))
            if self.input_transform:
                negative = self.input_transform(negative)
            negatives.append(negative)

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, [index, posIndex]+negIndices.tolist()

    def __len__(self):
        return len(self.queries)
