################################################################################
# CSE 151B: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin
# Winter 2022
################################################################################

from copy import deepcopy
import matplotlib.pyplot as plt
from simplejson import dump
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import nltk
import os
import pickle

import caption_utils 
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model




# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__lr = config_data['experiment']['learning_rate']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.
        self.__best_model_sd = {}
        # Init Model
        self.__model = get_model(config_data, self.__vocab)
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr = self.__lr)
        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__best_model_sd = state_dict['model']
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model.encoder = self.__model.encoder.cuda().float()
            self.__model.decoder = self.__model.decoder.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        min_loss = float('inf')
        patience = 0
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            # print('Epoch: ', epoch)
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            
            if val_loss < min_loss:
                patience = 0
                min_loss = val_loss
                self.__best_model_sd = self.__model.state_dict()
            else: patience +=1
            # print('Train Loss:', train_loss)
            # print('Validation Loss: ', val_loss)
            
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            if patience == 2: break
        self.__save_model()

    # TODO: Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        self.__model.encoder.train()
        self.__model.decoder.train()
        training_loss = 0
        batch_num = 0
        # Iterate over the data, implement the training function
        for i, (images, captions, _) in enumerate(self.__train_loader):
            batch_num += 1
            if batch_num % 500 == 0: print('Batch:', batch_num)
            images = images.cuda()
            captions = captions.cuda()
            self.__optimizer.zero_grad()
            output = self.__model(images, captions).view(-1, (len(self.__vocab)))
            loss = self.__criterion(output, captions.view(-1))
            training_loss += loss.item()
            loss.backward()
            self.__optimizer.step()            
        return training_loss/batch_num  

    # TODO: Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.__model.encoder.eval()
        self.__model.decoder.eval()
        val_loss = 0
        batch_num = 0
        with torch.no_grad():
            for i, (images, captions, _) in enumerate(self.__val_loader):
                batch_num += 1
                images = images.cuda()
                captions = captions.cuda()
                output = self.__model(images, captions).view(-1, (len(self.__vocab)))
                loss = self.__criterion(output, captions.view(-1))
                val_loss += loss.item()  
        return val_loss/batch_num

    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self):
        self.__model.load_state_dict(self.__best_model_sd)
        self.__model.encoder.eval()
        self.__model.decoder.eval()
        self.__model.encoder.cuda()
        self.__model.decoder.cuda()
        test_loss = 0
        bleu1 = 0
        bleu4 = 0
        batch_num = 0
        scores = []
        idxs = []
        caps2img = {}

        max_len = self.__generation_config['max_length']
        deterministic = self.__generation_config['deterministic']
        temperature = self.__generation_config['temperature']
        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.__test_loader):
                batch_num += 1
                images = images.cuda()
                captions = captions.cuda()
                output = self.__model(images, captions).view(-1, (len(self.__vocab)))
                loss = self.__criterion(output, captions.view(-1))
                test_loss += loss.item()
                caps = self.__model.generate_captions(images, max_len, deterministic, temperature)
                bleu1avg = 0
                bleu4avg = 0
                for i, id in enumerate(img_ids):
                    all_caps = self.__coco_test.imgToAnns[id]
                    ground_truth_caps = [nltk.word_tokenize(c['caption'].lower()) for c in all_caps]
                    bleu1avg += caption_utils.bleu1(ground_truth_caps,caps[i])
                    bleu4avg += caption_utils.bleu4(ground_truth_caps,caps[i])
                    # bleu1 score, image id, generated caption, given captions
                    if id not in idxs:
                        idxs.append(id)
                        scores.append([caption_utils.bleu1(ground_truth_caps,caps[i]), id, caps[i], ground_truth_caps])
                bleu1avg /= len(img_ids)
                bleu4avg /= len(img_ids)
                bleu1 += bleu1avg
                bleu4 += bleu4avg
             
        # Sort img captions by bleu1score
        scores.sort(key=lambda y: y[0])
        best3 = scores[-3:]
        worst3 = scores[:3]
        # Replace bleu1score with file name of image
        for ind in best3:
            ind[0] = self.__coco_test.loadImgs(ind[1])[0]['file_name']
        for ind in worst3:
            ind[0] = self.__coco_test.loadImgs(ind[1])[0]['file_name']
        # Dump file paths of best/worst images with generated captions to .pkl file
        dump_path =  os.path.join(self.__experiment_dir, self.__name + '_test_images.pkl')
        dump_file = open(dump_path, 'wb')
        caps2img['best'] = best3
        caps2img['worst'] = worst3
        pickle.dump(caps2img, dump_file)
        dump_file.close()

        test_loss /= batch_num
        bleu1 /= batch_num
        bleu4 /= batch_num
        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(test_loss,bleu1,bleu4)
        self.__log(result_str)

        return test_loss, bleu1, bleu4

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__best_model_sd
        # model_dict = self.__encoder.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
