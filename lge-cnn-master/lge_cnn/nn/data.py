from torch.utils.data import Dataset
import torch
from torch import Tensor
import h5py
import numpy as np


class YMDatasetHDF5(Dataset):
    def __init__(self, filename, mode_in='uw', mode_out='trW_2x2', use_idx=True, output_normalization=1.0):
        self.filename = filename
        self.f = None
        with h5py.File(self.filename, 'r') as file:
            self.num_samples = file['u'].shape[0]
            self.dims = np.array(file['dims'])

        self.beta = None
        self.mode_in, self.mode_out = mode_in, mode_out
        self.use_idx = use_idx
        self.output_normalization = output_normalization

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
            Multiprocessing fix for hdf5 files based on
            https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        """
        if self.f is None:
            self.f = h5py.File(self.filename, 'r')
            self.beta = self.f['beta'][:]

        # input mode
        if self.mode_in == 'uw':
            # format for x: combine u and w including real/imag in single tensor
            # layout: (batch_dim, lattice, channels, matrix structure, real/imag)
            # channels includes D links and N_W wilson loops
            u = self.f['u'][idx]
            u = torch.stack((Tensor(u.real), Tensor(u.imag)), dim=-1)
            w = self.f['w'][idx]
            w = torch.stack((Tensor(w.real), Tensor(w.imag)), dim=-1)
            x = torch.cat((u, w), dim=1)
        elif self.mode_in == 'uw_legacy':
            u = self.f['u'][idx]
            u = torch.stack((Tensor(u.real), Tensor(u.imag)), dim=-1)
            w = self.f['w'][idx]
            w_shape = w.shape
            w_reduced = w[:, 0:(w_shape[1] // 2)]
            w = torch.stack((Tensor(w_reduced.real), Tensor(w_reduced.imag)), dim=-1)
            x = torch.cat((u, w), dim=1)
        elif self.mode_in == 'u':
            u = self.f['u'][idx]
            u = torch.stack((Tensor(u.real), Tensor(u.imag)), dim=-1)
            x = u
        elif self.mode_in == 'u_su2_real':
            uR = Tensor(self.f['u'][idx].real)
            uI = Tensor(self.f['u'][idx].imag)

            a1 = uR.select(dim=-1, index=0).select(dim=-1, index=0)
            a2 = uI.select(dim=-1, index=0).select(dim=-1, index=0)
            a3 = uR.select(dim=-1, index=1).select(dim=-1, index=0)
            a4 = uI.select(dim=-1, index=1).select(dim=-1, index=0)

            u = torch.stack((a1, a2, a3, a4), dim=-1)
            x = u
        else:
            # put other possible input modes here
            print("Unknown mode_in for YMDatasetHDF5.")

        # output modes
        if self.mode_out.startswith('trW'):
            y = self.output_normalization * self.f[self.mode_out][idx].real
        elif self.mode_out in ['trP', 'QP', 'QC']:
            y = self.output_normalization * self.f[self.mode_out][idx]
        else:
            # put other possible output modes here
            print("Unknown mode_out for YMDatasetHDF5.")

        if self.use_idx:
            return x, y, idx
        else:
            return x, y

    def get_beta(self, idx):
        if self.f is None:
            self.f = h5py.File(self.filename, 'r')
        beta = self.f['beta'][:]
        return beta[idx]

    def close(self):
        if self.f is not None:
            self.f.close()

class CombinedFPActionDatasetHDF5(Dataset):
    def __init__(self, filenames, mode_in='u', mode_out='action', rescale_data=False, load_from='file'):
        if isinstance(filenames, list):
            self.filenames = filenames
        elif isinstance(filenames, str):
            self.filenames = [filenames]
        else:
            raise Exception("Argument `filenames` must either be list or str.")

        self.datasets = [FPActionDatasetHDF5(fn, mode_in, mode_out, rescale_data, load_from) for fn in self.filenames]
        self.lens = [len(ds) for ds in self.datasets]
        self.num_samples = sum(self.lens)
        self.acc_lens = [0] + list(np.cumsum(self.lens, dtype=int))

        # check dimensions of each dataset
        self.dims = self.datasets[0].dims
        for ds in self.datasets:
            if (ds.dims != self.dims).all():
                raise Exception("Dimensions mismatch between datasets.")
            
        for fn in self.filenames:
            #print(f"Loading dataset {fn}.")
            pass

    def close(self):
        for ds in self.datasets:
            ds.close()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx >= self.acc_lens[-1]:
            raise Exception("Sample index out of range.")

        sample_idx = None
        ds_idx = None
        for i in range(len(self.acc_lens)-1):
            if idx < self.acc_lens[i+1]:
                ds_idx = i
                sample_idx = idx - self.acc_lens[i]
                break
        
        if ds_idx is None or sample_idx is None:
            raise Exception("Something went wrong in selecting the dataset index in CombinedFPActionDatasetHDF5.")
        else:
            return self.datasets[ds_idx][sample_idx]

    def get_data(self, key):
        data = [ds.get_data(key) for ds in self.datasets]
        return np.concatenate(data)

class FPActionDatasetHDF5(Dataset):
    def __init__(self, filename, mode_in='u', mode_out='action', rescale_data=False, load_from='file'):
        self.filename = filename
        self.f = None
        self.beta = None
        self.mode_in, self.mode_out = mode_in, mode_out
        self.rescale_data = rescale_data
        self.load_from = load_from

        with h5py.File(self.filename, 'r') as file:
            self.num_samples = file['u'].shape[0]
            self.dims = np.array(file.attrs['dims'])

            # determine constants for rescaling S to [-1, +1]
            if rescale_data:
                self.a = 2 / (file.attrs['S_max'] - file.attrs['S_min'])
                self.b = - (file.attrs['S_max'] + file.attrs['S_min']) / (file.attrs['S_max'] - file.attrs['S_min'])
            else:
                self.a = 1.0
                self.b = 0.0

            self.S_var = file.attrs['S_var']
            self.DS_var = file.attrs['DS_var']
            
            # load dataset into memory (GPU)
            if load_from != 'file':
                self.U = torch.view_as_real(torch.tensor(file['u'][:]))
                self.S = torch.tensor(file['S'][:])
                self.DS = torch.view_as_real(torch.tensor(file['DS'][:]))
                self.B = torch.tensor(file['B'][:])

                if load_from == 'gpu':
                    self.U = self.U.cuda()
                    self.S = self.S.cuda()
                    self.DS = self.DS.cuda()
                    self.B = self.B.cuda()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.load_from == 'cpu' or self.load_from == 'gpu':
            return self.getitem_memory(idx)
        elif self.load_from == 'file':
            return self.getitem_hdf5(idx)
        else:
            raise Exception(f"Unknown method to load data (load_from): {self.load_from}")

    def close(self):
        # try to close hdf5 file
        if self.f is not None:
            self.f.close()
            del self.f
        
        # remove reference to hdf5 file
        self.f = None

        # remove references to arrays
        self.U = None
        self.S = None
        self.DS = None
        self.B = None

    def getitem_memory(self, idx):
        # input mode
        if self.mode_in == 'u':
            u = self.U[idx]
        else:
            print("Unknown mode_in for FPActionDatasetHDF5.")
        
        # output modes
        y_der = None
        if self.mode_out == 'action':
            y = self.a * self.S[idx] + self.b
        elif self.mode_out == 'action+derivative':
            y = self.a * self.S[idx] + self.b
            y_der = self.a * self.DS[idx]
        elif self.mode_out == 'norm_action+derivative':
            y = self.a * self.S[idx] / self.B[idx] + self.b
            y_der = self.a * self.DS[idx] / self.B[idx]
        else:
            # put other possible output modes here
            print("Unknown mode_out for FPActionDatasetHDF5.")

        if y_der is not None:
            return u, (y, y_der)
        else:
            return u, y

    def getitem_hdf5(self, idx):
        """
            Multiprocessing fix for hdf5 files based on
            https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        """
        if self.f is None:
            self.f = h5py.File(self.filename, 'r')

        # input mode
        if self.mode_in == 'u':
            u = self.f['u'][idx]
            u = torch.view_as_real(torch.tensor(u))
        else:
            # put other possible input modes here
            print("Unknown mode_in for FPActionDatasetHDF5.")

        # output modes
        y_der = None
        if self.mode_out == 'action':
            y = self.a * torch.tensor(self.f['S'][idx]) + self.b
        elif self.mode_out == 'action+derivative':
            y = self.a * torch.tensor(self.f['S'][idx]) + self.b
            y_der = self.a * torch.view_as_real(torch.tensor(self.f['DS'][idx]))
        elif self.mode_out == 'norm_action+derivative':
            y = self.a * torch.tensor(self.f['S'][idx] / self.f['B'][idx]) + self.b
            y_der = self.a * torch.view_as_real(torch.tensor(self.f['DS'][idx] / self.f['B'][idx]))
        else:
            # put other possible output modes here
            print("Unknown mode_out for FPActionDatasetHDF5.")

        if y_der is not None:
            return u, (y, y_der)
        else:
            return u, y
    
    def get_data(self, key):
        with h5py.File(self.filename, 'r') as file:
            return file[key][:]


class NewCombinedFPActionDatasetHDF5(Dataset):
    def __init__(self, filenames, load_from='file'):
        if isinstance(filenames, list):
            self.filenames = filenames
        elif isinstance(filenames, str):
            self.filenames = [filenames]
        else:
            raise Exception("Argument `filenames` must either be list or str.")

        self.datasets = [NewFPActionDatasetHDF5(fn, load_from) for fn in self.filenames]
        self.lens = [len(ds) for ds in self.datasets]
        self.num_samples = sum(self.lens)
        self.acc_lens = [0] + list(np.cumsum(self.lens, dtype=int))

        # check dimensions of each dataset
        self.dims = self.datasets[0].dims
        for ds in self.datasets:
            if (ds.dims != self.dims).all():
                raise Exception("Dimensions mismatch between datasets.")

    def close(self):
        for ds in self.datasets:
            ds.close()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx >= self.acc_lens[-1]:
            raise Exception("Sample index out of range.")

        sample_idx = None
        ds_idx = None
        for i in range(len(self.acc_lens)-1):
            if idx < self.acc_lens[i+1]:
                ds_idx = i
                sample_idx = idx - self.acc_lens[i]
                break
        
        if ds_idx is None or sample_idx is None:
            raise Exception("Something went wrong in selecting the dataset index in CombinedFPActionDatasetHDF5.")
        else:
            return self.datasets[ds_idx][sample_idx]

    def get_data(self, key):
        data = [ds.get_data(key) for ds in self.datasets]
        return np.concatenate(data)


class NewFPActionDatasetHDF5(Dataset):
    def __init__(self, filename, load_from='file'):
        self.filename = filename
        self.f = None
        self.beta = None
        self.load_from = load_from

        with h5py.File(self.filename, 'r') as file:
            self.num_samples = file['u'].shape[0]
            self.dims = np.array(file.attrs['dims'])
            
            # load dataset into memory (GPU)
            if load_from != 'file':
                self.U = torch.view_as_real(torch.tensor(file['u'][:]))
                self.S = torch.tensor(file['S_FP'][:])
                self.DS = torch.view_as_real(torch.tensor(file['DS_FP'][:]))
                self.B = torch.tensor(file['B'][:])

                if load_from == 'gpu':
                    self.U = self.U.cuda()
                    self.S = self.S.cuda()
                    self.DS = self.DS.cuda()
                    self.B = self.B.cuda()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.load_from == 'cpu' or self.load_from == 'gpu':
            return self.getitem_memory(idx)
        elif self.load_from == 'file':
            return self.getitem_hdf5(idx)
        else:
            raise Exception(f"Unknown method to load data (load_from): {self.load_from}")

    def close(self):
        # try to close hdf5 file
        if self.f is not None:
            self.f.close()
            del self.f
        
        # remove reference to hdf5 file
        self.f = None

        # remove references to arrays
        self.U = None
        self.S = None
        self.DS = None
        self.B = None

    def getitem_memory(self, idx):
        u = self.U[idx]
        y = self.S[idx]
        y_der = self.DS[idx]
        return u, (y, y_der)

    def getitem_hdf5(self, idx):
        """
            Multiprocessing fix for hdf5 files based on
            https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        """
        if self.f is None:
            self.f = h5py.File(self.filename, 'r')

        u = self.f['u'][idx]
        u = torch.view_as_real(torch.tensor(u))
        y = torch.tensor(self.f['S_FP'][idx])
        y_der = torch.view_as_real(torch.tensor(self.f['DS_FP'][idx]))

        return u, (y, y_der)
    
    def get_data(self, key):
        with h5py.File(self.filename, 'r') as file:
            return file[key][:]