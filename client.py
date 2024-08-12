from collections import OrderedDict
from typing import Dict
from flwr.common import NDArrays, Scalar, GetParametersIns, GetParametersRes, Status, Code, Parameters, FitIns, FitRes, \
    EvaluateRes, EvaluateIns

import torch
from torch.utils.data import Dataset, DataLoader
import flwr as fl

from model import SimpleCNN, MLP, train, test, RNNModel, testRNN, trainRNN, LSTM
from typing import List
from attacks import label_flipping_attack, targeted_label_flipping_attack, gan_attack, partial_dataset_for_GAN_attack
from attacks import mpaf_attack_nn, mpaf_attack_sklearn
from dataset import get_data_numpy

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import numpy as np
import warnings
from logging import INFO
from flwr.common.logger import log
import xgboost as xgb
from omegaconf import DictConfig

import time


def generate_client_fn(traindataset_list: List[Dataset], valdataset_list: List[Dataset], num_classes: int, model: str,
                       cfg: DictConfig):
    """Return a function that can be used by the VirtualClientEngine.

    to spawn a FlowerClient with client id `cid`.
    """

    def client_fn_lgr(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())

        # Returns a normal FLowerClient that will use the cid-th train/val
        # dataloaders as it's local data.
        return FlowerClientLGR(
            traindataset=traindataset_list[int(cid)],
            valdataset=valdataset_list[int(cid)],
            num_classes=num_classes,
            num_features=28 * 28  # TODO configurable?
        ).to_client()

    # Control logic for other models
    # return the function to spawn client
    return client_fn_lgr


class FlowerClientLGR(fl.client.NumPyClient):
    '''Define a Flower Client'''

    def __init__(self, traindataset: Dataset, valdataset: Dataset, num_classes: int, num_features) -> None:
        super().__init__()

        # the dataloaders that point to the data associated to this client
        self.traindataset = traindataset
        self.valdataset = valdataset

        # a model that is randomly initialised at first
        self.model = LogisticRegression()
        self.num_classes = num_classes
        self.num_features = num_features

        self.model.classes_ = np.array([i for i in range(num_classes)])
        self.model.coef_ = np.zeros((num_classes, self.num_features))
        if self.model.fit_intercept:
            self.model.intercept_ = np.zeros((num_classes,))
        self.attack_type = None
        self.is_malicious = False

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        """Sets the parameters of a sklean LogisticRegression model."""
        self.model.coef_ = parameters[0]
        if self.model.fit_intercept:
            self.model.intercept_ = parameters[1]

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""
        if self.model.fit_intercept:
            params = [self.model.coef_, self.model.intercept_]
        else:
            params = [self.model.coef_, ]
        if self.attack_type == "MPAF" and self.is_malicious:
            params = mpaf_attack_sklearn(params)
        return params

    def fit(self, parameters, config):
        """Train model received by the server (parameters) using the data.

        that belongs to this client. Then, send it back to the server.
        """
        self.attack_type = config["attack_type"]
        self.is_malicious = config["is_malicious"]
        # Poison the dataset if the client is malicious
        self.traindataset = applyAttacks(self.traindataset, config, model="LGR")

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        # fetch elements in the config sent by the server. Note that having a config
        # sent by the server each time a client needs to participate is a simple but
        # powerful mechanism to adjust these hyperparameters during the FL process. For
        # example, maybe you want clients to reduce their LR after a number of FL rounds.
        # or you want clients to do more local epochs at later stages in the simulation
        # you can control these by customising what you pass to `on_fit_config_fn` when
        # defining your strategy.
        penalty = config["penalty"]
        warm_start = config["warm_start"]
        epochs = config["local_epochs"]

        self.model.penalty = penalty
        self.model.warm_start = warm_start
        self.model.max_iter = epochs

        # do local training. This function is identical to what you might
        # have used before in non-FL projects. For more advance FL implementation
        # you might want to tweak it but overall, from a client perspective the "local
        # training" can be seen as a form of "centralised training" given a pre-trained
        # model (i.e. the model received from the server)
        trainloader = DataLoader(self.traindataset)
        # Convert to numpy data 
        X_train, y_train = get_data_numpy(trainloader)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_train, y_train)
            # print(f"Training finished for round {config['server_round']}")

        # Flower clients need to return three arguments: the updated model, the number
        # of examples in the client (although this depends a bit on your choice of aggregation
        # strategy), and a dictionary of metrics (here you can add any additional data, but these
        # are ideally small data structures)
        if config["defence"]:
            parameters = self.get_parameters({})
            # print("Testing--------------------------------------------------------------------------------")
            # time.sleep(60)
            loss, _, metrics = self.evaluate(parameters, config)
            metrics["loss"] = loss
            metrics["is_malicious"] = config["is_malicious"]
            return parameters, len(trainloader), metrics
        return self.get_parameters({}), len(X_train), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        valloader = DataLoader(self.valdataset)
        X_test, y_test = get_data_numpy(valloader)

        y_pred_prob = self.model.predict_proba(X_test)
        y_pred = self.model.predict(X_test)

        loss = log_loss(y_test, y_pred_prob)
        accuracy = self.model.score(X_test, y_test)

        # Precision, Recall, F1_score
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred, labels=list(range(10)))

        return float(loss), len(X_test), {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
                                          "confusion_matrix": conf_matrix}


def applyAttacks(trainset: Dataset, config, model: str = None) -> Dataset:
    # NOTE: this attack ratio is different, This is for number of samples to attack.
    ## The one in the config file is to select number of malicious clients

    if config["attack_type"] == "TLF":
        if config["is_malicious"]:
            print("----------------------------------Dataset Attacked------------------------------")
            return targeted_label_flipping_attack(trainset=trainset, attack_ratio=1.0)
    elif config["attack_type"] == "GAN":
        if config["is_malicious"]:
            print("----------------------------------Dataset Attacked------------------------------")
            return gan_attack(trainset=trainset)  # Change this if the program crashes
        # LGR model needs samples for all labels
        if model != "LGR":
            return partial_dataset_for_GAN_attack(trainset=trainset)
    elif config["attack_type"] == "MPAF":
        if config["is_malicious"]:
            print("----------------------------------Model Attacked------------------------------")
            return trainset
    else:
        if config["is_malicious"]:
            print("----------------------------------Dataset Attacked------------------------------")
            return label_flipping_attack(dataset=trainset, num_classes=10, attack_ratio=1.0)

    return trainset
