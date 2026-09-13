from argparse import ArgumentParser

import torch
from lge_cnn.nn import *
import numpy as np
import pytorch_lightning as pl
from math import sqrt
import scipy
from tqdm import tqdm
import pickle
from time import time

"""
    (Non-equivariant) Activation functions
"""


def get_activation(a_type):
    if a_type == 'relu':
        return torch.nn.ReLU()
    elif a_type == 'leakyrelu':
        return torch.nn.LeakyReLU()
    elif a_type == 'tanh':
        return torch.nn.Tanh()
    elif a_type == 'sigmoid':
        return torch.nn.Sigmoid()
    elif a_type == 'none':
        return None
    else:
        print("Option {} for activation function unimplemented.".format(a_type))
        return None


"""
    L-CNN model definition using LConvBilin modules
"""


class LConvBilinNet(pl.core.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        # number of lattice sites
        self.sites = np.prod(hparams.dims)

        # validation loss for progress bar
        self.vloss = 0.0

        # torch module lists
        self.convs = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()

        # add convolutions and pooling layer
        if hasattr(hparams, 'symmetric'):
            use_symmetric = hparams.symmetric
        else:
            use_symmetric = False

        conv_ch = list(hparams.conv_channels)
        conv_ch.insert(0, len(hparams.dims) * (len(hparams.dims) - 1) // 2)
        for i in range(len(conv_ch) - 1):
            conv = LConvBilin(dims=hparams.dims,
                              kernel_size=hparams.conv_kernel_size[i],
                              dilation=hparams.conv_dilation[i],
                              n_in=conv_ch[i],
                              n_out=conv_ch[i + 1],
                              nc=hparams.nc,
                              init_w=hparams.init_weight_factor,
                              use_symmetric=use_symmetric)

            self.convs.append(conv)

        # tracing layer
        self.tr = LTrace(hparams.dims)

        # input size of first linear layer
        linear_sizes = list(hparams.linear_sizes)
        linear_sizes.insert(0, 2 * conv_ch[-1])

        # output size of last layer
        linear_sizes.append(1)

        # add linear layers
        for i in range(len(linear_sizes) - 1):
            linear = torch.nn.Linear(linear_sizes[i], linear_sizes[i + 1], bias=True)
            self.linears.append(linear)
            if i < len(linear_sizes) - 2:
                act = get_activation(self.hparams.activation)
                if act is not None:
                    self.linears.append(act)

        self.out_mode = hparams.out_mode

        # output normalization
        if hasattr(hparams, 'output_norm'):
            output_norm = hparams.output_norm
        else:
            output_norm = 1.0

        # datasets
        self.train_dataset = YMDatasetHDF5(self.hparams.train_path,
                                           mode_in='uw_legacy', mode_out=self.out_mode, use_idx=False,
                                           output_normalization=output_norm)
        self.val_dataset = YMDatasetHDF5(self.hparams.val_path,
                                         mode_in='uw_legacy', mode_out=self.out_mode , use_idx=False,
                                         output_normalization=output_norm)
        self.test_dataset = YMDatasetHDF5(self.hparams.test_path,
                                          mode_in='uw_legacy', mode_out=self.out_mode, use_idx=False,
                                          output_normalization=output_norm)

        # check if dimensions match dataset
        for dataset in [self.train_dataset, self.val_dataset, self.test_dataset]:
            if isinstance(dataset, YMDatasetHDF5):
                if len(dataset.dims) != len(hparams.dims):
                    raise ValueError("Dimension mismatch between model ({}) and dataset ({})!".format(hparams.dims,
                                                                                                      dataset.dims))

                if (dataset.dims != hparams.dims).all():
                    raise ValueError("Dimension mismatch between model ({}) and dataset ({})!".format(hparams.dims,
                                                                                                      dataset.dims))
            else:
                print("Warning: cannot determine dimensions of dataset.")

    def forward(self, x):
        # store batch size
        batch_dim = x.shape[0]

        # apply GCMConv layers
        # x stays (batch_dim, lattice, channels, matrix structure)
        for i, conv in enumerate(self.convs):
            x = conv(x)

        # take trace
        # x becomes (batch_dim, lattice, channels)
        x = self.tr(x)

        x = x.view(batch_dim, self.sites, -1)
        # x = x[:, :, :, 0]

        # option to average over lattice sites
        if self.hparams.global_average:
            # x becomes (batch_dim, 2 * channels)
            x = torch.mean(x, dim=1)
        else:
            # x is (batch_dim, lattice sites, 2 * channels)
            # combine batch_dim and lattice sites into single dimension
            x = x.view(batch_dim * self.sites, -1)

        for i, layer in enumerate(self.linears):
            x = layer(x)

        # restore shape after linear layers
        if not self.hparams.global_average:
            x = x.view(batch_dim, self.sites, -1)

        return x

    """
        Prediction and testing
    """

    def evaluate(self, data='test', cuda=True):
        if data == 'train':
            dataset = self.train_dataset
            dataloader = self.train_dataloader()
        elif data == 'val':
            dataset = self.val_dataset
            dataloader = self.val_dataloader()
        elif data == 'test':
            dataset = self.test_dataset
            dataloader = self.test_dataloader()
        else:
            print("Unknown dataset.")
            return None

        X = []
        Y_pred = []
        Y_true = []

        dataset.use_idx = True

        with torch.no_grad():
            for x, y, idx in dataloader:
                beta = dataset.get_beta(idx)
                if cuda:
                    x = x.cuda()

                output = self(x)

                X.append(beta)
                Y_pred.append(output.detach().cpu().numpy())

                if self.hparams.global_average:
                    y = torch.mean(y, dim=1)
                y = y.view(output.shape)
                Y_true.append(y.detach().cpu().numpy())

        dataset.use_idx = False

        X = np.array(X).flatten()
        Y_pred = np.array(Y_pred)
        Y_true = np.array(Y_true)

        # combine batches
        num_batches = Y_pred.shape[0]
        batch_size = Y_pred.shape[1]
        rest_shape = Y_pred.shape[2:]

        Y_pred = Y_pred.reshape((num_batches * batch_size, *rest_shape))
        Y_true = Y_true.reshape((num_batches * batch_size, *rest_shape))

        return X, Y_pred, Y_true

    def mse(self, global_average=True, data='test', cuda=True):
        X, Y_pred, Y_true = self.evaluate(data, cuda)

        if global_average:
            Y_pred = np.mean(Y_pred, axis=1)
            Y_true = np.mean(Y_true, axis=1)

        mse_value = np.mean((Y_pred - Y_true).flatten() ** 2)

        return mse_value

    def update_dims(self, new_dims):
        if len(new_dims) != len(self.hparams.dims):
            raise ValueError("Cannot change lattice dimensions, only lattice size!")

        self.hparams.dims = new_dims
        self.sites = np.prod(new_dims)

        for conv in self.convs:
            conv.update_dims(new_dims)

        self.tr.update_dims(new_dims)

    """
        Additional methods
    """

    def close(self):
        self.train_dataset.close()
        self.val_dataset.close()
        self.test_dataset.close()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    """
        pytorch_lightning methods
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # architecture
        parser.add_argument('--dims', type=int, nargs='+')
        parser.add_argument('--nc', type=int, default=2)

        parser.add_argument('--global_average', action='store_true')
        parser.add_argument('--conv_channels', type=int, nargs='+')
        parser.add_argument('--conv_kernel_size', type=int, nargs='+')
        parser.add_argument('--conv_dilation', type=int, nargs='+')
        parser.add_argument('--init_weight_factor', type=float, default=1.0)
        parser.add_argument('--symmetric', action='store_true', default=False)

        parser.add_argument('--linear_sizes', type=int, nargs='*', default=[])
        parser.add_argument('--activation', type=str, default='relu')

        # datasets
        parser.add_argument('--train_path', type=str)
        parser.add_argument('--val_path', type=str)
        parser.add_argument('--test_path', type=str)
        parser.add_argument('--out_mode', type=str)
        parser.add_argument('--output_norm', type=float, default=1.0)
        parser.add_argument('--num_workers', type=int, default=0)

        # optimizer (AMSGradW)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--batch_size', type=int, default=10)
        parser.add_argument('--amsgrad', action='store_true', default=True)

        # lr scheduler (CosineAnnealingWarmRestarts)
        parser.add_argument('--use_scheduling', action='store_true')
        parser.add_argument('--T_0', type=int, default=10)
        parser.add_argument('--T_mult', type=int, default=1)
        parser.add_argument('--eta_min', type=float, default=0)

        return parser

    def get_progress_bar_dict(self):

        # call .item() only once but store elements without graphs
        running_train_loss = self.trainer.running_loss.mean()
        avg_training_loss = running_train_loss.cpu().item() if running_train_loss is not None else float('NaN')
        lr = self._scheduler.get_lr()[0] if self._scheduler is not None else self.hparams.lr

        tqdm_dict = {
            'loss': '{:.2E}'.format(avg_training_loss),
            'val_loss': '{:.2E}'.format(self.vloss),
            'lr': '{:.2E}'.format(lr)
        }

        if self.trainer.truncated_bptt_steps is not None:
            tqdm_dict['split_idx'] = self.trainer.split_idx

        if self.trainer.logger is not None and self.trainer.logger.version is not None:
            tqdm_dict['v_num'] = self.trainer.logger.version

        return tqdm_dict

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_dataset,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_dataset,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test_dataset,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=self.hparams.num_workers)

    def configure_optimizers(self):
        self._optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                            weight_decay=self.hparams.weight_decay, amsgrad=self.hparams.amsgrad)

        return_dict = {'optimizer': self._optimizer}

        self._scheduler = None

        if self.hparams.use_scheduling:
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self._optimizer,
                T_0=self.hparams.T_0,
                T_mult=self.hparams.T_mult,
                eta_min=self.hparams.eta_min,
                last_epoch=-1
            )

        if self._scheduler is not None:
            return_dict['lr_scheduler'] = self._scheduler

        return return_dict

    def loss(self, x, y):
        output = self(x).flatten()

        if self.hparams.global_average:
            y = torch.mean(y, dim=1)

        y = y.flatten()

        loss = torch.nn.functional.mse_loss(output, y)

        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)
        logs = {'loss': loss}

        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        self.vloss = avg_loss

        return {'avg_val_loss': avg_loss, 'log': logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)

        return {'val_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


"""
    Baseline CNN model
"""


class BaselineNet(pl.core.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # output normalization (optional)
        if hasattr(hparams, 'output_norm'):
            self.output_norm = hparams.output_norm
        else:
            self.output_norm = 1.0

        # input mode
        if hasattr(hparams, 'in_mode'):
            self.mode_in = hparams.in_mode
        else:
            self.mode_in = 'uw'

        # output mode
        self.out_mode = hparams.out_mode

        # number of lattice sites
        self.sites = np.prod(hparams.dims)
        self.dims = hparams.dims

        # validation loss for progress bar
        self.vloss = 0.0

        # torch module lists
        self.convs = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()

        # add convolutions
        # channel multiplication factor
        d = len(hparams.dims)
        if self.mode_in == 'uw':
            # links (U), plaquettes (W), daggered plaquettes (W^t)
            self.nch_factor = 2 * hparams.nc ** 2 * (d + 2 * d * (d-1) // 2)
        elif self.mode_in == 'uw_legacy':
            # links (U) and plaquettes (W)
            self.nch_factor = 2 * hparams.nc ** 2 * (d + d * (d-1) // 2)
        elif self.mode_in == 'u':
            # links (U)
            self.nch_factor = 2 * hparams.nc ** 2 * d
        else:
            raise NotImplementedError("Unknown in_mode {}".format(self.mode_in))

        conv_ch = list(hparams.conv_channels)
        conv_ch.insert(0, self.nch_factor)
        for i in range(len(conv_ch) - 1):
            conv = CConv2d(in_channels=conv_ch[i],
                           out_channels=conv_ch[i + 1],
                           kernel_size=hparams.conv_kernel_size[i],
                           bias=True)

            self.convs.append(conv)
            self.convs.append(get_activation(hparams.activation))

        # input size of first linear layer
        linear_sizes = list(hparams.linear_sizes)
        linear_sizes.insert(0, conv_ch[-1])

        # output size of last layer
        linear_sizes.append(1)

        # add linear layers
        for i in range(len(linear_sizes) - 1):
            linear = torch.nn.Linear(linear_sizes[i], linear_sizes[i + 1], bias=True)
            self.linears.append(linear)
            if i < len(linear_sizes) - 2:
                act = get_activation(hparams.activation)
                if act is not None:
                    self.linears.append(act)


        # datasets
        self.train_dataset = YMDatasetHDF5(self.hparams.train_path,
                                           mode_in=self.mode_in, mode_out=self.out_mode, use_idx=False,
                                           output_normalization=self.output_norm)
        self.val_dataset = YMDatasetHDF5(self.hparams.val_path,
                                         mode_in=self.mode_in, mode_out=self.out_mode, use_idx=False,
                                         output_normalization=self.output_norm)
        self.test_dataset = YMDatasetHDF5(self.hparams.test_path,
                                          mode_in=self.mode_in, mode_out=self.out_mode, use_idx=False,
                                          output_normalization=self.output_norm)

    def forward(self, x):
        # store batch size
        batch_dim = x.shape[0]

        # change shape
        x = x.view(batch_dim, *self.dims,  self.nch_factor)
        x = x.permute(0, 3, 1, 2)

        # apply CNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x)

        # option to average over lattice sites
        if self.hparams.global_average:
            x = torch.mean(x, dim=[2, 3])
            x = x.view(batch_dim, -1)
        else:
            # combine batch_dim and lattice sites into single dimension
            x = x.view(batch_dim * self.sites, -1)

        for i, layer in enumerate(self.linears):
            x = layer(x)

        # restore shape after linear layers
        if not self.hparams.global_average:
            x = x.view(batch_dim, self.sites, -1)

        return x

    """
        Prediction and testing
    """

    def evaluate(self, data='test', cuda=True):
        if data == 'train':
            dataset = self.train_dataset
            dataloader = self.train_dataloader()
        elif data == 'val':
            dataset = self.val_dataset
            dataloader = self.val_dataloader()
        elif data == 'test':
            dataset = self.test_dataset
            dataloader = self.test_dataloader()
        else:
            print("Unknown dataset.")
            return None

        X = []
        Y_pred = []
        Y_true = []

        dataset.use_idx = True

        with torch.no_grad():
            for x, y, idx in dataloader:
                beta = dataset.get_beta(idx)
                if cuda:
                    x = x.cuda()

                output = self(x)

                X.append(beta)
                Y_pred.append(output.detach().cpu().numpy())

                if self.hparams.global_average:
                    y = torch.mean(y, dim=1)
                y = y.view(output.shape)
                Y_true.append(y.detach().cpu().numpy())

        dataset.use_idx = False

        X = np.array(X).flatten()
        Y_pred = np.array(Y_pred)
        Y_true = np.array(Y_true)

        # combine batches
        num_batches = Y_pred.shape[0]
        batch_size = Y_pred.shape[1]
        rest_shape = Y_pred.shape[2:]

        Y_pred = Y_pred.reshape((num_batches * batch_size, *rest_shape))
        Y_true = Y_true.reshape((num_batches * batch_size, *rest_shape))

        return X, Y_pred, Y_true

    def mse(self, global_average=True, data='test', cuda=True):
        X, Y_pred, Y_true = self.evaluate(data, cuda)

        if global_average:
            Y_pred = np.mean(Y_pred, axis=1)
            Y_true = np.mean(Y_true, axis=1)

        mse_value = np.mean((Y_pred - Y_true).flatten() ** 2)

        return mse_value

    def update_dims(self, new_dims):
        if len(new_dims) != len(self.hparams.dims):
            raise ValueError("Cannot change lattice dimensions, only lattice size!")

        self.hparams.dims = new_dims
        self.dims = new_dims
        self.sites = np.prod(new_dims)

    """
        Additional methods
    """

    def close(self):
        self.train_dataset.close()
        self.val_dataset.close()
        self.test_dataset.close()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    """
        pytorch_lightning methods
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # architecture
        parser.add_argument('--dims', type=int, nargs='+')
        parser.add_argument('--nc', type=int, default=2)

        parser.add_argument('--global_average', action='store_true', default=True)
        parser.add_argument('--conv_channels', type=int, nargs='+')
        parser.add_argument('--conv_kernel_size', type=int, nargs='+')

        parser.add_argument('--linear_sizes', type=int, nargs='*', default=[])
        parser.add_argument('--activation', type=str)

        # datasets
        parser.add_argument('--train_path', type=str)
        parser.add_argument('--val_path', type=str)
        parser.add_argument('--test_path', type=str)
        parser.add_argument('--out_mode', type=str)
        parser.add_argument('--in_mode', type=str, default='uw')

        # optimizer (AMSGradW)
        parser.add_argument('--lr', type=float, default=3e-2)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--batch_size', type=int, default=50)
        parser.add_argument('--amsgrad', action='store_true', default=True)

        # lr scheduler (CosineAnnealingWarmRestarts)
        parser.add_argument('--use_scheduling', action='store_true', default=False)
        parser.add_argument('--T_0', type=int, default=10)
        parser.add_argument('--T_mult', type=int, default=1)
        parser.add_argument('--eta_min', type=float, default=0)

        return parser

    def get_progress_bar_dict(self):
        # call .item() only once but store elements without graphs
        running_train_loss = self.trainer.running_loss.mean()
        avg_training_loss = running_train_loss.cpu().item() if running_train_loss is not None else float('NaN')
        lr = self._scheduler.get_lr()[0] if self._scheduler is not None else self.hparams.lr

        tqdm_dict = {
            'loss': '{:.2E}'.format(avg_training_loss),
            'val_loss': '{:.2E}'.format(self.vloss),
            'lr': '{:.2E}'.format(lr)
        }

        if self.trainer.truncated_bptt_steps is not None:
            tqdm_dict['split_idx'] = self.trainer.split_idx

        if self.trainer.logger is not None and self.trainer.logger.version is not None:
            tqdm_dict['v_num'] = self.trainer.logger.version

        return tqdm_dict

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_dataset,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=True, num_workers=0)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_dataset,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=0)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test_dataset,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=0)

    def configure_optimizers(self):
        self._optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                            weight_decay=self.hparams.weight_decay, amsgrad=self.hparams.amsgrad)

        return_dict = {'optimizer': self._optimizer}

        self._scheduler = None

        if self.hparams.use_scheduling:
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self._optimizer,
                T_0=self.hparams.T_0,
                T_mult=self.hparams.T_mult,
                eta_min=self.hparams.eta_min,
                last_epoch=-1
            )

        if self._scheduler is not None:
            return_dict['lr_scheduler'] = self._scheduler

        return return_dict

    def loss(self, x, y):
        output = self(x).flatten()

        if self.hparams.global_average:
            y = torch.mean(y, dim=1)

        y = y.flatten()

        loss = torch.nn.functional.mse_loss(output, y)

        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)
        logs = {'loss': loss}

        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        self.vloss = avg_loss

        return {'avg_val_loss': avg_loss, 'log': logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)

        return {'val_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


"""
    New LCNN model definiton using separate LConv and LBilin
"""


class LCNN(pl.core.LightningModule):
    def __init__(self, hparams):
        super(LCNN, self).__init__()

        self.hparams = hparams

        # number of lattice sites
        self.sites = np.prod(hparams.dims)

        # validation loss for progress bar
        self.vloss = 0.0

        # L1 regularization
        if hasattr(hparams, 'L1'):
            self.L1 = hparams.L1
        else:
            self.L1 = 0.0

        # torch module lists
        self.convs = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()

        conv_inter_ch = list(hparams.conv_inter_channels)
        conv_ch = list(hparams.conv_channels)
        conv_ch.insert(0, 2 * len(hparams.dims) * (len(hparams.dims) - 1) // 2)
        for i in range(len(conv_ch) - 1):
            conv = LConvBilin2(dims=hparams.dims,
                               kernel_size=hparams.conv_kernel_size[i],
                               dilation=hparams.conv_dilation[i],
                               n_in=conv_ch[i],
                               n_inter=conv_inter_ch[i],
                               n_out=conv_ch[i + 1],
                               nc=hparams.nc,
                               use_unit_elements=True,
                               extended=False)

            self.convs.append(conv)

        # tracing layer
        self.tr = LTrace(hparams.dims)

        # input size of first linear layer
        linear_sizes = list(hparams.linear_sizes)
        linear_sizes.insert(0, 2 * conv_ch[-1])

        # output size of last layer
        linear_sizes.append(1)

        # add linear layers
        for i in range(len(linear_sizes) - 1):
            linear = torch.nn.Linear(linear_sizes[i], linear_sizes[i + 1], bias=True)
            self.linears.append(linear)
            if i < len(linear_sizes) - 2:
                act = get_activation(hparams.activation)
                if act is not None:
                    self.linears.append(act)

        self.out_mode = hparams.out_mode

        # datasets
        self.train_dataset = YMDatasetHDF5(self.hparams.train_path,
                                           mode_in='uw', mode_out=self.out_mode, use_idx=False)
        self.val_dataset = YMDatasetHDF5(self.hparams.val_path,
                                         mode_in='uw', mode_out=self.out_mode , use_idx=False)
        self.test_dataset = YMDatasetHDF5(self.hparams.test_path,
                                          mode_in='uw', mode_out=self.out_mode, use_idx=False)

    def forward(self, x):
        # store batch size
        batch_dim = x.shape[0]

        # apply GCMConv layers
        # x stays (batch_dim, lattice, channels, matrix structure)
        for i, conv in enumerate(self.convs):
            x = conv(x)

        # take trace
        # x becomes (batch_dim, lattice, channels)
        x = self.tr(x)

        x = x.view(batch_dim, self.sites, -1)
        # x = x[:, :, :, 0]

        # option to average over lattice sites
        if self.hparams.global_average:
            # x becomes (batch_dim, 2 * channels)
            x = torch.mean(x, dim=1)
        else:
            # x is (batch_dim, lattice sites, 2 * channels)
            # combine batch_dim and lattice sites into single dimension
            x = x.view(batch_dim * self.sites, -1)

        for i, layer in enumerate(self.linears):
            x = layer(x)

        # restore shape after linear layers
        if not self.hparams.global_average:
            x = x.view(batch_dim, self.sites, -1)

        return x

    """
        Prediction and testing
    """

    def evaluate(self, data='test', cuda=True):
        if data == 'train':
            dataset = self.train_dataset
            dataloader = self.train_dataloader()
        elif data == 'val':
            dataset = self.val_dataset
            dataloader = self.val_dataloader()
        elif data == 'test':
            dataset = self.test_dataset
            dataloader = self.test_dataloader()
        else:
            print("Unknown dataset.")
            return None

        X = []
        Y_pred = []
        Y_true = []

        dataset.use_idx = True

        with torch.no_grad():
            for x, y, idx in dataloader:
                beta = dataset.get_beta(idx)
                if cuda:
                    x = x.cuda()

                output = self(x)

                X.append(beta)
                Y_pred.append(output.detach().cpu().numpy())

                if self.hparams.global_average:
                    y = torch.mean(y, dim=1)
                y = y.view(output.shape)
                Y_true.append(y.detach().cpu().numpy())

        dataset.use_idx = False

        X = np.array(X).flatten()
        Y_pred = np.array(Y_pred)
        Y_true = np.array(Y_true)

        # combine batches
        num_batches = Y_pred.shape[0]
        batch_size = Y_pred.shape[1]
        rest_shape = Y_pred.shape[2:]

        Y_pred = Y_pred.reshape((num_batches * batch_size, *rest_shape))
        Y_true = Y_true.reshape((num_batches * batch_size, *rest_shape))

        return X, Y_pred, Y_true

    def mse(self, global_average=True, data='test', cuda=True):
        X, Y_pred, Y_true = self.evaluate(data, cuda)

        if global_average:
            Y_pred = np.mean(Y_pred, axis=1)
            Y_true = np.mean(Y_true, axis=1)

        mse_value = np.mean((Y_pred - Y_true).flatten() ** 2)

        return mse_value

    def update_dims(self, new_dims):
        if len(new_dims) != len(self.hparams.dims):
            raise ValueError("Cannot change lattice dimensions, only lattice size!")

        self.hparams.dims = new_dims
        self.sites = np.prod(new_dims)

        for conv in self.convs:
            conv.update_dims(new_dims)

        self.tr.update_dims(new_dims)

    """
        Additional methods
    """

    def close(self):
        self.train_dataset.close()
        self.val_dataset.close()
        self.test_dataset.close()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    """
        pytorch_lightning methods
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # architecture
        parser.add_argument('--dims', type=int, nargs='+')
        parser.add_argument('--nc', type=int, default=2)

        parser.add_argument('--global_average', action='store_true')
        parser.add_argument('--conv_channels', type=int, nargs='+')
        parser.add_argument('--conv_inter_channels', type=int, nargs='+')
        parser.add_argument('--conv_kernel_size', type=int, nargs='+')
        parser.add_argument('--conv_dilation', type=int, nargs='+')

        parser.add_argument('--linear_sizes', type=int, nargs='*', default=[])
        parser.add_argument('--activation', type=str, default='relu')

        # datasets
        parser.add_argument('--train_path', type=str)
        parser.add_argument('--val_path', type=str)
        parser.add_argument('--test_path', type=str)
        parser.add_argument('--out_mode', type=str)
        parser.add_argument('--num_workers', type=int, default=0)

        # optimizer (AMSGradW)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--batch_size', type=int, default=10)
        parser.add_argument('--amsgrad', action='store_true', default=True)

        # lr scheduler (CosineAnnealingWarmRestarts)
        parser.add_argument('--use_scheduling', action='store_true')
        parser.add_argument('--T_0', type=int, default=10)
        parser.add_argument('--T_mult', type=int, default=1)
        parser.add_argument('--eta_min', type=float, default=0)

        return parser

    def get_progress_bar_dict(self):

        # call .item() only once but store elements without graphs
        running_train_loss = self.trainer.running_loss.mean()
        avg_training_loss = running_train_loss.cpu().item() if running_train_loss is not None else float('NaN')
        lr = self._scheduler.get_lr()[0] if self._scheduler is not None else self.hparams.lr

        tqdm_dict = {
            'loss': '{:.2E}'.format(avg_training_loss),
            'val_loss': '{:.2E}'.format(self.vloss),
            'lr': '{:.2E}'.format(lr)
        }

        if self.trainer.truncated_bptt_steps is not None:
            tqdm_dict['split_idx'] = self.trainer.split_idx

        if self.trainer.logger is not None and self.trainer.logger.version is not None:
            tqdm_dict['v_num'] = self.trainer.logger.version

        return tqdm_dict

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_dataset,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_dataset,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test_dataset,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=self.hparams.num_workers)

    def configure_optimizers(self):
        self._optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                            weight_decay=self.hparams.weight_decay, amsgrad=self.hparams.amsgrad)

        return_dict = {'optimizer': self._optimizer}

        self._scheduler = None

        if self.hparams.use_scheduling:
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self._optimizer,
                T_0=self.hparams.T_0,
                T_mult=self.hparams.T_mult,
                eta_min=self.hparams.eta_min,
                last_epoch=-1
            )

        if self._scheduler is not None:
            return_dict['lr_scheduler'] = self._scheduler

        return return_dict

    def loss(self, x, y):
        output = self(x).flatten()

        if self.hparams.global_average:
            y = torch.mean(y, dim=1)

        y = y.flatten()

        loss = torch.nn.functional.mse_loss(output, y)

        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)

        if self.L1 != 0.0:
            # Add L1 regularization
            L1_reg = torch.tensor(0., requires_grad=True).cuda()
            for name, param in self.named_parameters():
                if 'weight' in name:
                    L1_reg = L1_reg + torch.norm(param, 1)
            loss = loss + self.L1 * L1_reg

        logs = {'loss': loss}

        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        self.vloss = avg_loss

        return {'avg_val_loss': avg_loss, 'log': logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)

        return {'val_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
    
    
    
"""
    An L-CNN model for learning actions with derivatives
"""

class FPActionModel(pl.core.LightningModule):
    def __init__(self, dims, nc, conv_channels, conv_kernel_size, conv_dilation, linear_sizes, activation, init_weight_factor, 
                train_path, val_path, test_path, out_mode, group_derivative_weight, lr, batch_size,
                use_only_real_part=True, initial_linear_size=None, input_type='plaq', final_fit='const', fit_every_epoch=False,
                loss_func='l1',
                **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # number of lattice sites
        self.sites = int(np.prod(dims))

        # validation loss for progress bar
        self.vloss = 0.0

        # Plaq layer
        if input_type == 'plaq':
            self.plaq = Plaq(dims, nc, orientation=-1)
            n_input_channels = self.plaq.get_number_of_channels()
        elif input_type == 'clover':
            self.plaq = PlaqClover(dims, nc)
            n_input_channels = self.plaq.get_number_of_channels()
        else:
            raise Exception(f"Unkown input_type: {input_type}")
        
        # initial gauge equivariant linear layer
        if initial_linear_size is not None:
            self.initial_llin = LLin(dims, n_input_channels, initial_linear_size, nc)
        else:
            self.initial_llin = None

        # torch module lists
        self.convs = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()

        # add convolutions
        conv_ch = list(conv_channels)

        # add first layer with correct number of channels from Plaq layer or from initial linear layer
        if initial_linear_size is not None:
            conv_ch.insert(0, initial_linear_size)
        else:
            conv_ch.insert(0, n_input_channels)

        #assert conv_ch[-1] == 1, "The last channel must be size 1."
        for i in range(len(conv_ch) - 1):
            conv = LConvBilin(dims=dims,
                              kernel_size=conv_kernel_size[i],
                              dilation=conv_dilation[i],
                              n_in=conv_ch[i],
                              n_out=conv_ch[i + 1],
                              nc=nc,
                              init_w=init_weight_factor,
                              use_symmetric=True)

            self.convs.append(conv)

        # tracing layer
        self.tr = LTrace(dims)

        # input size of first linear layer
        linear_sizes = list(linear_sizes)

        # option for choosing between real part of trace, or both real and imaginary parts
        if use_only_real_part:
            linear_sizes.insert(0, conv_ch[-1]) # real part
        else:
            linear_sizes.insert(0, 2*conv_ch[-1]) # real and imaginary part

        # output size of last layer
        linear_sizes.append(1)

        # add linear layers
        for i in range(len(linear_sizes) - 1):
            linear = torch.nn.Linear(linear_sizes[i], linear_sizes[i + 1], bias=True)
            # initialize bias to zero, or not?
            torch.nn.init.constant_(linear.bias, 0.0)

            self.linears.append(linear)
            if i < len(linear_sizes) - 2:
                act = get_activation(self.hparams.activation)
                if act is not None:
                    self.linears.append(act)

        # manual affine scaling
        self.affine_a = 1.0
        self.affine_b = 0.0

        self.out_mode = out_mode
        self.global_average = True

        # ah layer
        self.ah = LAH(dims, nc)

        self.timings = {
            'plaq': [],
            'tr': [],
            'convs': [],
            'linears': []
        }

        self.call_counter = 0
        self.report_step = -1 # set to -1 to turn off
    
    def forward(self, u):
        # store batch size
        batch_dim = u.shape[0]

        # compute plaquettes
        t0 = time()
        w = self.plaq(u)
        self.timings['plaq'].append(time() - t0)

        # pack u and w into x
        x = repack_x(u, w)

        # apply initial gauge equivariant linear layer
        if self.initial_llin is not None:
            x = self.initial_llin(x)

        # apply LConfBilin layers
        # x stays (batch_dim, lattice, channels, matrix structure)
        t0 = time()
        for i, conv in enumerate(self.convs):
            x = conv(x)
        self.timings['convs'].append(time() - t0)

        # take trace
        # x becomes (batch_dim, lattice, channels)
        t0 = time()
        x = self.tr(x)
        self.timings['tr'].append(time() - t0)

        # take real part
        if self.hparams.use_only_real_part:
            x = x[..., 0]

        # reformat to apply linear layers at every point
        x = x.view(batch_dim * self.sites, -1)

        # apply linear layers
        t0 = time()
        for i, layer in enumerate(self.linears):
            x = layer(x)
        self.timings['linears'].append(time() - t0)

        # reformat back to original
        x = x.view(batch_dim, self.sites, -1)

        if self.global_average:
            # perform global sum (or mean ..)
            x = torch.sum(x, dim=1)

            # put channels and real/imag into a single channel
            x = x.view(batch_dim, -1)
        else:
            # put channels and real/imag into a single channel
            x = x.view(batch_dim*self.sites, -1)

        
        if self.global_average:
            # apply affine layer
            x = self.affine_a * x + self.affine_b
        else:
            # restore lattice
            x = x.view(batch_dim, self.sites, -1)

            # apply affine layer
            x = self.affine_a * x + self.affine_b / self.sites

        self.call_counter += 1
        if self.call_counter % self.report_step == 0 and self.report_step > 0:
            print("*** Timings")
            for key in self.timings:
                print(f"{key}: {np.mean(self.timings[key][-self.report_step:-1]):3.3f}s")

        return x


    def group_derivative(self, u):
        with torch.enable_grad():
            # make model differentiable w.r.t. input
            u = u.clone().detach().requires_grad_(True)
            f = self(u)
            df = torch.autograd.grad(f, u, grad_outputs=torch.ones_like(f), retain_graph=True, create_graph=True)[0]

        dfU = 0.5 * torch.transpose(torch.view_as_complex(df), dim0=-1, dim1=-2).conj()
        UdfU = torch.einsum('bxdij, bxdjk -> bxdik', torch.view_as_complex(u), dfU)
        UdfU_ah = self.ah(torch.view_as_real(UdfU))
        Df = -UdfU_ah

        return f, Df

    """
        Freezing/unfreezing weights
    """
    def freeze_lcnn_modules(unfreeze=False):
        pass

    """
        Prediction and testing
    """
    def locality_map(self, data='test', samples=1):
        """
            
        """

    def evaluate(self, data='test', amount=1.0, override_batch_size=None, shuffle=False):
        """
            Returns predictions and true values for a given dataset.
        """

        # override batch_size for evaluation
        if override_batch_size is not None:
            print(f"Temporarily overriding batch_size from {self.hparams.batch_size} to {override_batch_size}.")
            original_batch_size = self.hparams.batch_size
            self.hparams.batch_size = override_batch_size

        if data == 'train':
            dataset = self.train_dataset
            dataloader = self.train_dataloader(shuffle=shuffle)
        elif data == 'val':
            dataset = self.val_dataset
            dataloader = self.val_dataloader(shuffle=shuffle)
        elif data == 'test':
            dataset = self.test_dataset
            dataloader = self.test_dataloader(shuffle=shuffle)
        else:
            print("Unknown dataset.")
            return None

        n_samples = len(dataset)
        batch_size = self.hparams.batch_size
        n_batches = int(np.ceil(n_samples / batch_size))
        n_red_batches = int(amount * n_batches)

        Y_pred = []
        Y_true = []

        DY_pred = []
        DY_true = []

        index = 0

        with tqdm(total=n_red_batches) as pbar:
            with torch.no_grad():
                for u, y_batch in dataloader:
                    if self.out_mode == 'action':
                        y_true = y_batch
                        y_pred = self(u.cuda())
                    else:
                        y_true, dy_true = y_batch
                        y_pred, dy_pred = self.group_derivative(u.cuda())

                        DY_pred.append(dy_pred.detach().cpu().numpy())
                        DY_true.append(dy_true.detach().cpu().numpy())

                    Y_pred.append(y_pred.detach().cpu().numpy())
                    Y_true.append(y_true.detach().cpu().numpy())

                    index += 1
                    pbar.update()

                    if index >= n_red_batches:
                        break
        
        Y_pred = np.concatenate(Y_pred, axis=0).flatten()
        Y_true = np.concatenate(Y_true, axis=0).flatten()

        if self.out_mode == 'action+derivative' or self.out_mode == 'norm_action+derivative':
            DY_pred = np.concatenate(DY_pred, axis=0).flatten()
            DY_true = np.concatenate(DY_true, axis=0).flatten()

        # set original batch_size
        if override_batch_size is not None:
            print(f"Setting batch_size back to {original_batch_size}.")
            self.hparams.batch_size = original_batch_size

        if self.out_mode == 'action':
            return Y_pred, Y_true
        else:
            return (Y_pred, DY_pred), (Y_true, DY_true)
    
    def mse(self, data='test'):
        """
            Computes MSE based on action values
        """
        pred, true = self.evaluate(data)

        if self.out_mode == 'action':
            Y_pred, Y_true = pred, true
            mse_value = np.mean((Y_pred - Y_true).flatten() ** 2)
            return mse_value
        else:
            (Y_pred, DY_pred), (Y_true, DY_true) = pred, true
            mse_value = np.mean((Y_pred - Y_true).flatten() ** 2)
            mse_value_der = np.mean((DY_pred - DY_true).flatten() ** 2)
            return mse_value, mse_value_der

    def perform_final_layer_fit(self, data='train', amount=1.0, override_batch_size=None):
        # set to initial values
        self.affine_a = 1.0
        self.affine_b = 0.0

        # compute predictions for dataset (default: 'train')
        if self.out_mode == 'action':
            Y_pred, Y_true = self.evaluate(data, amount, override_batch_size, shuffle=True)
        else:
            (Y_pred, DY_pred), (Y_true, DY_true) = self.evaluate(data, amount, override_batch_size, shuffle=True)

        # perform fit
        if self.hparams.final_fit == 'const':
            def func_const(x, c0):
                return x + c0
            
            fit_params, _ = scipy.optimize.curve_fit(func_const, Y_pred, Y_true)

            # set affine parameters
            self.affine_a = 1.0
            self.affine_b = fit_params[0]

        elif self.hparams.final_fit == 'affine':
            def func_affine(x, c0, c1):
                return c1*x + c0
            
            fit_params, _ = scipy.optimize.curve_fit(func_affine, Y_pred, Y_true)

            # set affine parameters
            self.affine_a = fit_params[1]
            self.affine_b = fit_params[0]
        else:
            raise Exception(f"Unknown final_fit type: {self.hparams.final_fit}")

    def update_dims(self, new_dims):
        if len(new_dims) != len(self.hparams.dims):
            raise ValueError("Cannot change lattice dimensions, only lattice size!")

        self.hparams.dims = new_dims
        self.sites = np.prod(new_dims)

        self.plaq.update_dims(new_dims)
        for conv in self.convs:
            conv.update_dims(new_dims)

        self.tr.update_dims(new_dims)

    """
        pickle-based saving
    """

    def save(self, filepath):
        self.close()
        pickle.dump(self, open(filepath, 'wb'))
        self.setup()

    @staticmethod
    def load(filepath):
        model = pickle.load(open(filepath, 'rb'))
        model.setup()
        return model

    """
        Additional methods
    """

    def close(self):
        self.train_dataset.close()
        self.val_dataset.close()
        self.test_dataset.close()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    """
        pytorch_lightning methods
    """

    def setup(self, stage=None):
        # datasets
        self.train_dataset = CombinedFPActionDatasetHDF5(self.hparams.train_path, mode_in='u', mode_out=self.hparams.out_mode,
                                                 rescale_data=self.hparams.rescale_data, load_from=self.hparams.load_from)
        self.val_dataset = CombinedFPActionDatasetHDF5(self.hparams.val_path, mode_in='u', mode_out=self.hparams.out_mode,
                                               rescale_data=self.hparams.rescale_data, load_from=self.hparams.load_from)
        self.test_dataset = CombinedFPActionDatasetHDF5(self.hparams.test_path, mode_in='u', mode_out=self.hparams.out_mode,
                                                rescale_data=self.hparams.rescale_data, load_from=self.hparams.load_from)

        # check if dimensions match dataset
        for dataset in [self.train_dataset, self.val_dataset, self.test_dataset]:
            if isinstance(dataset, FPActionDatasetHDF5) or isinstance(dataset, CombinedFPActionDatasetHDF5):
                if len(dataset.dims) != len(self.hparams.dims):
                    raise ValueError("Dimension mismatch between model ({}) and dataset ({})!".format(self.hparams.dims,
                                                                                                      dataset.dims))

                if (dataset.dims != self.hparams.dims).all():
                    raise ValueError("Dimension mismatch between model ({}) and dataset ({})!".format(self.hparams.dims,
                                                                                                      dataset.dims))
            else:
                print("Warning: cannot determine dimensions of dataset.")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = parent_parser.add_argument_group("FPActionModel")

        # architecture
        #parser.add_argument('--dims', type=int, nargs='+')
        #parser.add_argument('--nc', type=int, default=2)

        #parser.add_argument('--conv_channels', type=int, nargs='+')
        #parser.add_argument('--conv_kernel_size', type=int, nargs='+')
        #parser.add_argument('--conv_dilation', type=int, nargs='+')
        #parser.add_argument('--init_weight_factor', type=float, default=1.0)

        # datasets
        #parser.add_argument('--train_path', type=str)
        #parser.add_argument('--val_path', type=str)
        #parser.add_argument('--test_path', type=str)
        #parser.add_argument('--out_mode', type=str)
        #parser.add_argument('--num_workers', type=int, default=0)

        # optimizer (AMSGradW)
        #parser.add_argument('--lr', type=float, default=3e-4)
        #parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--batch_size', type=int, default=10)

        return parser

    def prepare_data(self):
        pass

    def train_dataloader(self, shuffle=True):
        return torch.utils.data.DataLoader(dataset=self.train_dataset,
                                           batch_size=self.hparams.batch_size, pin_memory=False,
                                           shuffle=shuffle, num_workers=self.hparams.num_workers)

    def val_dataloader(self, shuffle=False):
        return torch.utils.data.DataLoader(dataset=self.val_dataset,
                                           batch_size=self.hparams.batch_size, pin_memory=False,
                                           shuffle=shuffle, num_workers=self.hparams.num_workers)

    def test_dataloader(self, shuffle=False):
        return torch.utils.data.DataLoader(dataset=self.test_dataset,
                                           batch_size=self.hparams.batch_size, pin_memory=False,
                                           shuffle=shuffle, num_workers=self.hparams.num_workers)

    def configure_optimizers(self):
        self._optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                            weight_decay=0.0, amsgrad=True)
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, mode='min', factor=0.33, patience=10)

        # lets try something else
        #self._optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        #self._optimizer = torch.optim.NAdam(self.parameters(), lr=self.hparams.lr)
        #self._optimizer = torch.optim.LBFGS(self.parameters(), lr=self.hparams.lr)
        return_dict = {'optimizer': self._optimizer, "lr_scheduler": self._scheduler, "monitor": "val_loss"}

        return return_dict

    def loss(self, u, data):
        if self.hparams.loss_func == 'l1':
            return self.l1_loss(u, data)
        elif self.hparams.loss_func == 'l1_log':
            return torch.log(self.l1_loss(u, data))
        elif self.hparams.loss_func == 'gauge_invariant':
            return self.invariant_loss(u, data)
        elif self.hparams.loss_func == 'gauge_invariant_sqrt':
            return torch.sqrt(self.invariant_loss(u, data))
        elif self.hparams.loss_func == 'gauge_invariant_log':
            return torch.log(self.invariant_loss(u, data))
        else:
            raise Exception(f"Unknown loss function: {self.hparams.loss_func}")
        
    def l1_loss(self, u, data):
        batch_dim = u.shape[0]
        if self.hparams.out_mode == 'action':
            y = data
            output = self(u).reshape(batch_dim, -1)
            y = y.reshape(batch_dim, -1)
            loss = torch.nn.functional.l1_loss(output, y)
        elif self.hparams.out_mode == 'action+derivative' or self.hparams.out_mode == 'norm_action+derivative':
            y, yder = data
            ypred, ypred_der = self.group_derivative(u)
            loss_s = torch.nn.functional.l1_loss(ypred.reshape(batch_dim, -1), y.reshape(batch_dim, -1))
            loss_ds = torch.nn.functional.l1_loss(ypred_der.reshape(batch_dim, -1), yder.reshape(batch_dim, -1))
            w = self.hparams.group_derivative_weight

            if w < 1.0:
                loss = sqrt(1-w**2) * loss_s + w * loss_ds # maybe take log?
            else:
                loss = w * loss_ds
        else:
            raise Exception(f"Unknown out_mode: {self.hparams.out_mode}")

        return loss

    def invariant_loss(self, u, data):
        batch_dim = u.shape[0]
        n_sites = u.shape[1]
        n_dims = u.shape[2]
        if self.hparams.out_mode == 'action':
            y = data
            output = self(u).reshape(batch_dim, -1)
            y = y.reshape(batch_dim, -1)
            loss = torch.nn.functional.l1_loss(output, y)
        elif self.hparams.out_mode == 'action+derivative' or self.hparams.out_mode == 'norm_action+derivative':
            y, yder = data
            ypred, ypred_der = self.group_derivative(u)
            loss_s = torch.nn.functional.l1_loss(ypred.reshape(batch_dim, -1), y.reshape(batch_dim, -1))
            delta_der = yder - ypred_der
            loss_ds = complex_einsum('bxdij, bxdji -> ', delta_der, delta_der)[0] / (batch_dim * n_sites * n_dims)
            w = self.hparams.group_derivative_weight

            if w < 1.0:
                loss = sqrt(1-w**2) * loss_s + w * loss_ds # maybe take log?
            else:
                loss = w * loss_ds
        else:
            raise Exception(f"Unknown out_mode: {self.hparams.out_mode}")

        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)
        self.log("val_loss", loss)

        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss)
        self.vloss = avg_loss

        if self.hparams.fit_every_epoch:
            # perform affine fit and compute mean rel error
            self.perform_final_layer_fit(amount=1.0)
            (YP, DYP), (YT, DYT) = self.evaluate('val')
            err_Y = np.mean(np.abs((YP - YT) / YT))
            self.log("avg_rel_err", err_Y)
            print("******")
            print(f"Abs. rel. error (action values) after final layer fit ({self.hparams.final_fit}): {100*err_Y:3.3f}%")
            print("******")

            # reset affine map
            self.affine_a = 1.0
            self.affine_b = 0.0

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)
        self.log("test_loss", loss)
        return loss

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss)
        self.vloss = avg_loss



"""
    An L-CNN model for learning actions with derivatives
"""

class FPDerivativeModel(pl.core.LightningModule):
    def __init__(self, dims, nc, conv_channels, conv_kernel_size, conv_dilation, init_weight_factor, 
                train_path, val_path, test_path, lr, batch_size,
                input_type='plaq', loss_func='l1',
                **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # number of lattice sites
        self.sites = int(np.prod(dims))

        # validation loss for progress bar
        self.vloss = 0.0

        # Plaq layer
        if input_type == 'plaq':
            self.plaq = Plaq(dims, nc, orientation=-1)
            n_input_channels = self.plaq.get_number_of_channels()
        elif input_type == 'clover':
            self.plaq = PlaqClover(dims, nc)
            n_input_channels = self.plaq.get_number_of_channels()
        else:
            raise Exception(f"Unkown input_type: {input_type}")


        # torch module lists
        self.convs = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()

        # add convolutions
        conv_ch = list(conv_channels)

        # add first channel
        conv_ch.insert(0, n_input_channels)

        assert conv_ch[-1] == len(dims), "The last channel must be equal the number of dimensions."
        for i in range(len(conv_ch) - 1):
            conv = LConvBilin(dims=dims,
                              kernel_size=conv_kernel_size[i],
                              dilation=conv_dilation[i],
                              n_in=conv_ch[i],
                              n_out=conv_ch[i + 1],
                              nc=nc,
                              init_w=init_weight_factor,
                              use_symmetric=True)

            self.convs.append(conv)

        # ah layer
        self.ah = LAH(dims, nc)

        self.timings = {
            'plaq': [],
            'tr': [],
            'convs': [],
            'linears': []
        }

        self.call_counter = 0
        self.report_step = -1 # set to -1 to turn off
    
    def forward(self, u):
        # store batch size
        batch_dim = u.shape[0]

        # compute plaquettes
        t0 = time()
        w = self.plaq(u)
        self.timings['plaq'].append(time() - t0)

        # pack u and w into x
        x = repack_x(u, w)

        # apply LConfBilin layers
        # x stays (batch_dim, lattice, channels, matrix structure)
        t0 = time()
        for i, conv in enumerate(self.convs):
            x = conv(x)
        self.timings['convs'].append(time() - t0)

        # project onto lie algebra
        u, w = unpack_x(x, len(self.hparams.dims))
        w_ah = self.ah(w)

        return w_ah

    def evaluate(self, data='test', amount=1.0, override_batch_size=None, shuffle=False):
        """
            Returns predictions and true values for a given dataset.
        """

        # override batch_size for evaluation
        if override_batch_size is not None:
            print(f"Temporarily overriding batch_size from {self.hparams.batch_size} to {override_batch_size}.")
            original_batch_size = self.hparams.batch_size
            self.hparams.batch_size = override_batch_size

        if data == 'train':
            dataset = self.train_dataset
            dataloader = self.train_dataloader(shuffle=shuffle)
        elif data == 'val':
            dataset = self.val_dataset
            dataloader = self.val_dataloader(shuffle=shuffle)
        elif data == 'test':
            dataset = self.test_dataset
            dataloader = self.test_dataloader(shuffle=shuffle)
        else:
            print("Unknown dataset.")
            return None

        n_samples = len(dataset)
        batch_size = self.hparams.batch_size
        n_batches = int(np.ceil(n_samples / batch_size))
        n_red_batches = int(amount * n_batches)

        DY_pred = []
        DY_true = []

        index = 0

        with tqdm(total=n_red_batches) as pbar:
            with torch.no_grad():
                for u, y_batch in dataloader:
                    dy_true = y_batch[1]
                    dy_pred = self(u.cuda())

                    DY_pred.append(dy_pred.detach().cpu().numpy())
                    DY_true.append(dy_true.detach().cpu().numpy())

                    index += 1
                    pbar.update()

                    if index >= n_red_batches:
                        break
        
        DY_pred = np.concatenate(DY_pred, axis=0).flatten()
        DY_true = np.concatenate(DY_true, axis=0).flatten()

        # set original batch_size
        if override_batch_size is not None:
            print(f"Setting batch_size back to {original_batch_size}.")
            self.hparams.batch_size = original_batch_size
        
        return DY_pred, DY_true

    def update_dims(self, new_dims):
        if len(new_dims) != len(self.hparams.dims):
            raise ValueError("Cannot change lattice dimensions, only lattice size!")

        self.hparams.dims = new_dims
        self.sites = np.prod(new_dims)

        self.plaq.update_dims(new_dims)
        for conv in self.convs:
            conv.update_dims(new_dims)

        self.ah.update_dims(new_dims)

    """
        pickle-based saving
    """

    def save(self, filepath):
        self.close()
        pickle.dump(self, open(filepath, 'wb'))
        self.setup()

    @staticmethod
    def load(filepath):
        model = pickle.load(open(filepath, 'rb'))
        model.setup()
        return model

    """
        Additional methods
    """

    def close(self):
        self.train_dataset.close()
        self.val_dataset.close()
        self.test_dataset.close()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    """
        pytorch_lightning methods
    """

    def setup(self, stage=None):
        # datasets
        self.train_dataset = CombinedFPActionDatasetHDF5(self.hparams.train_path, mode_in='u', mode_out=self.hparams.out_mode,
                                                 rescale_data=self.hparams.rescale_data, load_from=self.hparams.load_from)
        self.val_dataset = CombinedFPActionDatasetHDF5(self.hparams.val_path, mode_in='u', mode_out=self.hparams.out_mode,
                                               rescale_data=self.hparams.rescale_data, load_from=self.hparams.load_from)
        self.test_dataset = CombinedFPActionDatasetHDF5(self.hparams.test_path, mode_in='u', mode_out=self.hparams.out_mode,
                                                rescale_data=self.hparams.rescale_data, load_from=self.hparams.load_from)

        # check if dimensions match dataset
        for dataset in [self.train_dataset, self.val_dataset, self.test_dataset]:
            if isinstance(dataset, FPActionDatasetHDF5) or isinstance(dataset, CombinedFPActionDatasetHDF5):
                if len(dataset.dims) != len(self.hparams.dims):
                    raise ValueError("Dimension mismatch between model ({}) and dataset ({})!".format(self.hparams.dims,
                                                                                                      dataset.dims))

                if (dataset.dims != self.hparams.dims).all():
                    raise ValueError("Dimension mismatch between model ({}) and dataset ({})!".format(self.hparams.dims,
                                                                                                      dataset.dims))
            else:
                print("Warning: cannot determine dimensions of dataset.")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = parent_parser.add_argument_group("FPActionModel")

        # architecture
        #parser.add_argument('--dims', type=int, nargs='+')
        #parser.add_argument('--nc', type=int, default=2)

        #parser.add_argument('--conv_channels', type=int, nargs='+')
        #parser.add_argument('--conv_kernel_size', type=int, nargs='+')
        #parser.add_argument('--conv_dilation', type=int, nargs='+')
        #parser.add_argument('--init_weight_factor', type=float, default=1.0)

        # datasets
        #parser.add_argument('--train_path', type=str)
        #parser.add_argument('--val_path', type=str)
        #parser.add_argument('--test_path', type=str)
        #parser.add_argument('--out_mode', type=str)
        #parser.add_argument('--num_workers', type=int, default=0)

        # optimizer (AMSGradW)
        #parser.add_argument('--lr', type=float, default=3e-4)
        #parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--batch_size', type=int, default=10)

        return parser

    def prepare_data(self):
        pass

    def train_dataloader(self, shuffle=True):
        return torch.utils.data.DataLoader(dataset=self.train_dataset,
                                           batch_size=self.hparams.batch_size, pin_memory=False,
                                           shuffle=shuffle, num_workers=self.hparams.num_workers)

    def val_dataloader(self, shuffle=False):
        return torch.utils.data.DataLoader(dataset=self.val_dataset,
                                           batch_size=self.hparams.batch_size, pin_memory=False,
                                           shuffle=shuffle, num_workers=self.hparams.num_workers)

    def test_dataloader(self, shuffle=False):
        return torch.utils.data.DataLoader(dataset=self.test_dataset,
                                           batch_size=self.hparams.batch_size, pin_memory=False,
                                           shuffle=shuffle, num_workers=self.hparams.num_workers)

    def configure_optimizers(self):
        self._optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                            weight_decay=0.0, amsgrad=True)
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, mode='min', factor=0.33, patience=10)

        # lets try something else
        #self._optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        #self._optimizer = torch.optim.NAdam(self.parameters(), lr=self.hparams.lr)
        #self._optimizer = torch.optim.LBFGS(self.parameters(), lr=self.hparams.lr)
        return_dict = {'optimizer': self._optimizer, "lr_scheduler": self._scheduler, "monitor": "val_loss"}

        return return_dict

    def loss(self, u, data):
        if self.hparams.loss_func == 'l1':
            return self.l1_loss(u, data)
        elif self.hparams.loss_func == 'l1_log':
            return torch.log(self.l1_loss(u, data))
        elif self.hparams.loss_func == 'gauge_invariant':
            return self.invariant_loss(u, data)
        elif self.hparams.loss_func == 'gauge_invariant_sqrt':
            return torch.sqrt(self.invariant_loss(u, data))
        elif self.hparams.loss_func == 'gauge_invariant_log':
            return torch.log(self.invariant_loss(u, data))
        else:
            raise Exception(f"Unknown loss function: {self.hparams.loss_func}")
        
    def l1_loss(self, u, data):
        batch_dim = u.shape[0]

        y, yder = data
        ypred_der = self(u)
        loss = torch.nn.functional.l1_loss(ypred_der.reshape(batch_dim, -1), yder.reshape(batch_dim, -1))

        return loss

    def invariant_loss(self, u, data):
        batch_dim = u.shape[0]
        n_sites = u.shape[1]
        n_dims = u.shape[2]

        y, yder = data
        ypred_der = self(u)
        delta_der = yder - ypred_der
        loss = complex_einsum('bxdij, bxdji -> ', delta_der, delta_der)[0] / (batch_dim * n_sites * n_dims)

        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)
        self.log("val_loss", loss)

        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss)
        self.vloss = avg_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)
        self.log("test_loss", loss)
        return loss

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss)
        self.vloss = avg_loss
