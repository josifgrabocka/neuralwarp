from distutils.command.config import config

from optimizer import Optimizer
from hyper_parameters import HyperParams
from dataset import Dataset
import rnn_models
import cnn_models
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append("warp/")

# get the configurations
hp = HyperParams()
config = hp.get_uniwarp_config(sys.argv)

# create the optimizer
dataset_folder = sys.argv[1]
dataset = Dataset()
dataset.load_multivariate(dataset_folder)

config['uniwarp:length']=dataset.series_length
config['dataset:num_channels']=dataset.num_channels
# create the model
model = None

if sys.argv[2] == 'SiameseRNN':
    model = rnn_models.SiameseRNN(config=config)
elif sys.argv[2] == 'WarpedSiameseRNN':
        model = rnn_models.WarpedSiameseRNN(config=config)
elif sys.argv[2] == 'CNNSim':
        model = cnn_models.CNNSim(config=config)
elif sys.argv[2] == 'CNNWarpedSim':
        model = cnn_models.CNNWarpedSim(config=config)

model.create_model()

print("This model has", model.num_model_parameters(), "parameters")

opt = Optimizer(config=config, dataset=dataset, sim_model=model)
opt.optimize()
