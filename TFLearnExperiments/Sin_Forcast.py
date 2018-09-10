import numpy as np
import tensorflow as tf
import matplotlib as mpl
import tflearn
from tflearn.layers.recurrent import lstm
mpl.use('Agg')
from matplotlib import pyplot as plt

HIDDEN_SIZE = 30
NUM_LAYERS = 2
TIMESTEPS = 10
TRAINING_STEPS = 10000
BATCH_SIZE = 32
TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01


