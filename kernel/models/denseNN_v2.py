from logging.config import valid_ident
from operator import index
from random import shuffle
from textwrap import indent
import torch
import pandas as pd
import numpy as np
import random
import logging
import csv
from torch import nn
from torch.optim import Adam
from typing import Tuple, List
from tqdm import tqdm
from collections import OrderedDict
from torch.serialization import default_restore_location


from kernel.networks.fully_connected import FCNetwork
from kernel.utils.checkpoints import save_checkpoint
from kernel.utils.logging import init_logging
from kernel.models.util import get_analytics
import os

class DenseNN(nn.Module):

    def __init__(self, input_dims: int, output_dims: int, depth:int, width:int, **kwargs):
        """
        dsfsd
        """
        super(DenseNN, self).__init__()
        layers = tuple([width for i in range(depth)])
        self.network = FCNetwork((input_dims, *layers, output_dims), output_activation=None, **kwargs)
        self.network_opt = Adam(self.network.parameters(), lr=kwargs["learning_rate"], weight_decay=kwargs['l2'])
        self.training_data = None
        self.validation_data = None
        self.test_data = None
        self.config = kwargs
        # self.loss = nn.MSELoss()
        self.loss = nn.L1Loss()

        self.network.to(self.config['device'])

        lambda1 = lambda epoch: self.config['decay_rate'] ** epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.network_opt , lr_lambda=lambda1)

        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.network_opt, base_lr=0.00001, max_lr=kwargs["learning_rate"],step_size_up=5,mode="exp_range",gamma=0.85)

        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.network_opt, T_0=20, T_mult=1, eta_min=0.0000001, last_epoch=-1)/


        self.validation_stats = []
        self.test_state = []

        if self.config['seed']:
            print("======================= seeded =====================", self.config['seed'])
            torch.manual_seed(self.config['seed'])
            torch.cuda.manual_seed(self.config['seed'])
            torch.cuda.manual_seed_all(self.config['seed'])  # if you are using multi-GPU.
            np.random.seed(self.config['seed'])  # Numpy module.
            random.seed(self.config['seed'])  # Python random module.
            torch.manual_seed(self.config['seed'])
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            os.environ['PYTHONHASHSEED'] = str(self.config['seed'])

        return

    def forward(self, x):
        return self.network(x)

    def load_data(self):
        # TODO ----------- remove yaw and target error
        x = pd.read_csv(self.config["data_dir"][0], index_col=0).to_numpy()
        y = pd.read_csv(self.config["data_dir"][1], index_col=0).to_numpy()

        print(x.shape)
       

        N = len(x)

        data = list(zip(x,y))
        random.shuffle(data)

        self.training_data = data[:int(0.8*N)]
        self.validation_data = data[int(0.8*N):]

        print(N)
        print(len(y))
        print(len(data))
        print(len(self.training_data))
        print(len(self.validation_data))
        input()
        # self.test_data = data[int(0.9*N):]
        return

    def train_model(self):

        init_logging(self.config)
        logging.info('Commencing training!')

        stats = OrderedDict()
        stats['loss'] = 0
        stats['lr'] = self.network_opt.param_groups[0]['lr']
        stats['validation_loss'] = 0

        valid_best = 100
  
        no_improvement_count = 0
        epoch = 0

        self.validate_model()


        while True:
        # for epoch in range(self.config["max_epoch"]):
            no_improvement_count += 1

            if epoch > self.config["max_epoch"] and self.config["mode"] != 'rollout-training':
                break

            if self.config["mode"] == 'rollout-training' and no_improvement_count == 10:
                break


            train_loader = torch.utils.data.DataLoader(self.training_data, num_workers=1, batch_size=self.config['batch_size'], shuffle=True)

            self.train()
            

            progress_bar = tqdm(train_loader, desc='| Epoch {:03d}'.format(epoch), leave=False, disable=False)

            for i, sample in enumerate(progress_bar):

                output = self(sample[0].float().to(self.config['device']))

                
                loss = self.loss(output, sample[1].float().to(self.config['device']))
                loss.backward()
                self.network_opt.step()
                self.network_opt.zero_grad()

                stats['loss'] += loss.cpu().item()

                progress_bar.set_postfix({key: '{:.4g}'.format(value / (i + 1)) for key, value in stats.items()},
                                     refresh=True)


            self.scheduler.step()


            
            stats['validation_loss'] = self.validate_model()
            print((stats['loss'])/(len(self.training_data)/self.config["batch_size"]),stats['validation_loss'], self.network_opt.param_groups[0]['lr'])

            if stats['validation_loss'] < valid_best:
                no_improvement_count = 0
                valid_best = stats['validation_loss'] 

            
            if self.config['mode'] != "gs":
                save_checkpoint(self.config, self, self.network_opt, epoch, stats['validation_loss'])
                logging.info('Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.4g}'.format(value / len(progress_bar)) for key, value in stats.items())))

                with open(f'{self.config["save_dir"]}_losses.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([stats['loss'], stats['validation_loss'] ])

            stats['loss'] = 0
            stats['validation_loss'] = 0


        if self.config['mode'] == "training":
            np.save(f'{self.config["save_dir"]}_stats.npy', self.validation_stats)


        return valid_best

    def validate_model(self):

        stats = OrderedDict()
        stats['validation_loss'] = 0
  

        self.eval()
  
        train_loader = torch.utils.data.DataLoader(self.validation_data, num_workers=1, batch_size=self.config['batch_size'], shuffle=True)
        progress_bar = tqdm(train_loader, desc='| VALIDATION', leave=False, disable=False)


        predictions = torch.Tensor()
        targets = torch.Tensor()

        for i, sample in enumerate(progress_bar):
            output = self(sample[0].float().to(self.config['device']))
            loss = self.loss(output, sample[1].float().to(self.config['device']))

            predictions = torch.cat((predictions, output.detach().cpu()), 0)
            targets = torch.cat((targets, sample[1].detach().cpu()), 0)

            stats['validation_loss'] += loss.cpu().item()

            progress_bar.set_postfix({key: '{:.4g}'.format(value / (i + 1)) for key, value in stats.items()},
                                    refresh=True)

        analaytics = get_analytics(predictions,targets)
        self.validation_stats.append(analaytics)
        
        return (stats['validation_loss'])/(len(self.validation_data)/self.config["batch_size"])

    def test_model(self):
        stats = OrderedDict()
        stats['validation_loss'] = 0
  

        self.eval()
        
        train_loader = torch.utils.data.DataLoader(self.test_data, num_workers=1, batch_size=self.config['batch_size'], shuffle=True)
        progress_bar = tqdm(train_loader, desc='| TEST', leave=False, disable=False)

        for i, sample in enumerate(progress_bar):
            output = self(sample[0].float().to(self.config['device']))
            s = sample[1].float().to(self.config['device'])

            print(output[0])
            print(s[0])
            input()


       
            loss = self.loss(output, s)
            
            stats['validation_loss'] += loss.item()

            progress_bar.set_postfix({key: '{:.4g}'.format(value / (i + 1)) for key, value in stats.items()},
                                    refresh=True)


        return((stats['validation_loss'])/(len(self.validation_data)/self.config["batch_size"]))


    def view_plot_foot_positions(self):

        self.eval()

        x = torch.Tensor(pd.read_csv("./data/sets/walking_cmd_x.csv", index_col=0).to_numpy())[100000:110000]
        y = torch.Tensor(pd.read_csv("./data/sets/walking_cmd_y.csv", index_col=0).to_numpy())[100000:110000]

        pred_y = self(torch.Tensor(x).float().to(self.config['device'])).detach().cpu()


        X = 0
        Y = 1
        Z = 2


        import matplotlib.pyplot as plt
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)

 
        
        c = ['r', 'r', 'r', 'r']
        for limb in range(0,4):

            x_true = y[:,X+limb*3]
            y_true = y[:,Y+limb*3]
            z_true = y[:,Z+limb*3]

            x_pred = pred_y[:,X+limb*3]
            y_pred = pred_y[:,Y+limb*3]
            z_pred = pred_y[:,Z+limb*3]

            ax1.plot(x_true, c[limb] )
            ax1.plot(x_pred, 'b')
            # ax11.plot(x[:,X+limb], "b")
            ax1.set_title('X')

            ax2.plot(y_true, c[limb])
            ax2.plot(y_pred, 'b')
            # ax22.plot(x[:,X+limb], "b")
            ax2.set_title('Y')


            ax3.plot(z_true, c[limb])
            ax3.plot(z_pred, 'b')
            # ax33.plot(x[:,X+limb], "b")
            ax3.set_title('Z')



        plt.show()


        return


    def view_plot_q_q_dot(self):

        self.eval()

        x = torch.Tensor(pd.read_csv("./data/sets/episode_joints_x.csv", index_col=0).to_numpy())
        y = torch.Tensor(pd.read_csv("./data/sets/episode_joints_y.csv", index_col=0).to_numpy())

        pred_y = self(torch.Tensor(x).float().to(self.config['device'])).detach().cpu()


        X = 0
        Y = 12
        Z = 2


        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)

        ax1t = ax1.twinx()
    
        
        #joint angles

        for joint in range(0,12):

            x_true = y[:,X+joint]
            v_true = y[:,Y+joint]
          

            x_pred = pred_y[:,X+joint]
            v_pred = pred_y[:,Y+joint]
            z_pred = pred_y[:,Z+joint]

            ax1.plot(x_true, 'r')
            ax1.plot(x_pred, 'g')
            ax1t.plot(x[:,0], "b")
            ax1t.plot(x[:,1], "b")
            ax1t.plot(x[:,2], "b")
            ax1t.plot(x[:,3], "b")
            ax1.set_title('X')

        plt.show()

        for limb in range(0,4):
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)

            ax1t = ax1.twinx()
            ax2t = ax2.twinx()
            ax3t = ax3.twinx()

            ax_a = [ax1,ax2,ax3]
            ax_t = [ax1t,ax2t,ax3t]

            for joint in range(0,3):

                v_true = y[:,Y+(limb*joint)]
                v_pred = pred_y[:,Y+(limb*joint)]
        

                ax_a[joint].plot(v_true, 'r')
                ax_a[joint].plot(v_pred, 'g')

                ax_t[joint].plot(x[:,0], "b")
                ax_t[joint].plot(x[:,1], "b")
                ax_t[joint].plot(x[:,2], "b")
                ax_t[joint].plot(x[:,3], "b")

                ax_a[joint].set_title('X')

            plt.show()

            # ax2.plot(y_true, 'r')
            # ax2.plot(y_pred, 'g')
            # ax22.plot(x[:,X+limb], "b")
            # ax2.set_title('Y')


            # ax3.plot(z_true, 'r')
            # ax3.plot(z_pred, 'g')
            # ax33.plot(x[:,X+limb], "b")
            # ax3.set_title('Z')



    def load_model(self):
        state_dict = torch.load(self.config['model_file'], map_location=lambda s, l: default_restore_location(s, 'cpu'))
        self.load_state_dict(state_dict['model'])
        return 