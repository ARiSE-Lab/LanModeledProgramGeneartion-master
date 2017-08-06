###############################################################################
# Author: Md Rizwan Parvez
# Project: LanModeledProgramGeneration
# Date Created: 4/7/2017
# Many codes are from Wasi Ahmad train.py
# File Description: This script contains code to train the model.
###############################################################################

import time, util, torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm
import math, os, shutil
import numpy as np

args = util.get_args()


class Train:
    """Train class that encapsulate all functionalities of the training procedure."""

    def __init__(self, model_f, model_b, dictionary, loss_f):
        self.dictionary = dictionary
        self.forward_model = model_f
        self.backward_model = model_b
        self.loss_f = loss_f
        self.criterion = getattr(nn, self.loss_f)(size_average=True) # nn.CrossEntropyLoss()  # Combines LogSoftMax and NLLoss in one single class
        self.num_directions = 2 if args.bidirection else 1
        self.forward_lr = self.backward_lr = args.lr

        # Adam optimizer is used for stochastic optimization
        self.forward_optimizer = optim.SGD(self.forward_model.parameters(), self.forward_lr)
        self.backward_optimizer = optim.SGD(self.backward_model.parameters(), self.backward_lr)

    def train_single_epoch(self, train_data_trimed, train_label_trimed , valid_data_trimed, valid_label_trimed, trainF, testF, epoch, best_perplexity = math.exp(100), direction = 'forward'): #train_batches, dev_batches
        
        try:

            epoch_start_time = time.time()

            #plot_losses = 
            self.train(train_data_trimed, train_label_trimed, epoch, trainF, direction)
            #print(plot_losses)
            #util.save_plot(plot_losses, args.save_path + 'training_loss_plot_epoch_{}.png'.format((epoch)))

            if direction=='forward':
                model = self.forward_model
                optimizer= self.forward_optimizer
                lr = self.forward_lr
            else:
                model = self.backward_model
                optimizer= self.backward_optimizer
                lr = self.backward_lr
          
            val_loss = util.evaluate(valid_data_trimed, valid_label_trimed , model, self.dictionary, self.criterion, epoch, testF, direction)

            ppl = math.exp(val_loss)

            print('-' * 89)
            print('| end of ', direction, ' epoch {:3d} | time: {:5.2f}s | valid loss {:.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, ppl ))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            is_best = ppl < best_perplexity
            best_perplexity = min(ppl, best_perplexity)
            filename  = os.path.join(args.log_dir, direction+'checkpoint.pth.tar')
            
            
            torch.save({'epoch':epoch + 1, 'state_dict': model.state_dict(), 'perplexity': ppl}, filename)    
            
            
            #os.system('python plot.py {} &'.format(args.log_dir))
            if is_best:
                print("saving as best model")
                shutil.copyfile(filename, os.path.join(args.log_dir, direction+'_model_best.pth.tar'))
            else:
                
                lr  *= args.lr_decay
                optimizer.param_groups[0]['lr'] = lr
                print("Decaying learning rate to %g" % lr)
            return best_perplexity
                
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    def train_epochs(self, train_data_trimed, train_label_trimed , valid_data_trimed, valid_label_trimed): #train_batches, dev_batches
        """Trains model for n_epochs epochs"""
        # Loop over epochs.
        best_val_loss = None

        # At any point you can hit Ctrl + C to break out of training early.
        try:

            best_perplexity_forward = best_perplexity_backward = math.exp(100)
            if args.resume:
                train_forward_F = open(os.path.join(args.log_dir, 'forward_train.csv'), 'a')
                test_forward_F  = open(os.path.join(args.log_dir, 'forward_test.csv'), 'a')
                train_backward_F = open(os.path.join(args.log_dir, 'backward_train.csv'), 'a')
                test_backward_F = open(os.path.join(args.log_dir, 'backward_test.csv'), 'a')
                if os.path.isfile(os.path.join(args.log_dir, 'forwardcheckpoint.pth.tar')) and os.path.isfile(os.path.join(args.log_dir, 'backwardcheckpoint.pth.tar')):
                    print("=> loading  checkpoints")
                    checkpoint_forward= torch.load(os.path.join(args.log_dir, 'forwardcheckpoint.pth.tar'))
                    checkpoint_backward= torch.load(os.path.join(args.log_dir, 'backwardcheckpoint.pth.tar'))
                    args.start_epoch = checkpoint_forward['epoch']
                    best_perplexity_forward = checkpoint_forward['perplexity']
                    best_perplexity_backward = checkpoint_backward['perplexity']
                    self.forward_model.load_state_dict(checkpoint_forward['state_dict'])
                    self.backward_model.load_state_dict(checkpoint_backward['state_dict'])
                    print("=> loaded checkpoint (epoch {})".format(args.start_epoch))
                else:
                    print("=> no checkpoint found")
            else:
                train_forward_F = open(os.path.join(args.log_dir, 'forward_train.csv'), 'w')
                test_forward_F  = open(os.path.join(args.log_dir, 'forward_test.csv'), 'w')
                train_backward_F = open(os.path.join(args.log_dir, 'backward_train.csv'), 'w')
                test_backward_F = open(os.path.join(args.log_dir, 'backward_test.csv'), 'w')
                

            print("===start training===")
            for epoch in range(args.start_epoch, args.nepochs + 1):
            #for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()

                #plot_losses = 
                #self.train(train_data_trimed, train_label_trimed, epoch, train_forward_F)
                best_perplexity_forward =  self.train_single_epoch( train_data_trimed, train_label_trimed , valid_data_trimed, valid_label_trimed, train_forward_F, test_forward_F, epoch, best_perplexity_forward, direction = 'forward')
                best_perplexity_backward =  self.train_single_epoch( train_data_trimed, train_label_trimed , valid_data_trimed, valid_label_trimed, train_backward_F, test_backward_F, epoch, best_perplexity_backward, direction = 'backward')
                #print(plot_losses)
                #util.save_plot(plot_losses, args.save_path + 'training_loss_plot_epoch_{}.png'.format((epoch)))

                val_loss = self.validate(valid_data_trimed, valid_label_trimed , self.forward_model, self.backward_model, epoch)
                ppl = math.exp(val_loss)

                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | bidirectional valid loss {:.2f} | '
                      'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                 val_loss, ppl ))
                print('-' * 89)
                
                

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    def save_checkpoint(args, state, is_best, filename):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(args.log_dir, 'model_best.pth.tar'))

    def train_(self, train_batches, dev_batches, epoch_no ):
        # Turn on training mode which enables dropout.
        self.model.train()

        start = time.time()
        plot_losses = []
        print_loss_total = 0
        plot_loss_total = 0
        best_dev_loss = -1
        last_best_dev_loss = -1

        num_batches = len(train_batches)
        print('epoch %d started' % epoch_no)

        for batch_no in range(num_batches):
            # Clearing out all previous gradient computations.
            self.optimizer.zero_grad()
            train_sentences1, train_sentences2, train_labels = util.instances_to_tensors(train_batches[batch_no],
                                                                                           self.dictionary)
            if args.cuda:
                train_sentences1 = train_sentences1.cuda()
                train_sentences2 = train_sentences2.cuda()
                train_labels = train_labels.cuda()

            assert train_sentences1.size(0) == train_sentences2.size(0)

            softmax_prob = self.model(train_sentences1, train_sentences2)
            loss = self.criterion(softmax_prob, train_labels)

            # Important if we are using nn.DataParallel()
            if loss.size(0) > 1:
                loss = torch.mean(loss)
            loss.backward()

            print_loss_total += loss.data[0]
            plot_loss_total += loss.data[0]

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
            clip_grad_norm(self.model.parameters(), args.clip)
            self.optimizer.step()

            if batch_no % args.print_every == 0 and batch_no > 0:
                print_loss_avg = print_loss_total / args.print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (
                    util.show_progress(start, batch_no / num_batches), batch_no,
                    batch_no / num_batches * 100, print_loss_avg))

            if batch_no % args.plot_every == 0 and batch_no > 0:
                plot_loss_avg = plot_loss_total / args.plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

            if batch_no % args.dev_every == 0 and batch_no > 0:
                dev_loss = self.validate(dev_batches)
                print('validation loss = %.4f' % dev_loss)
                if best_dev_loss == -1 or best_dev_loss > dev_loss:
                    best_dev_loss = dev_loss
                else:
                    # no improvement in validation loss, so apply learning rate decay
                    self.lr = self.lr * args.lr_decay
                    self.optimizer.param_groups[0]['lr'] = self.lr
                    print("Decaying learning rate to %g" % self.lr)

            if batch_no % args.save_every == 0 and batch_no > 0:
                if last_best_dev_loss == -1 or last_best_dev_loss > best_dev_loss:
                    last_best_dev_loss = best_dev_loss
                    util.save_model(self.model, last_best_dev_loss, epoch_no, 'model')

        return plot_losses

    def train(self, train_data_trimed, train_label_trimed, epoch, trainF, direction = 'forward'):
        # Turn on training mode which enables dropout.
        
        batch_loss = 0
        start_time = time.time()

        plot_losses = []
        print_loss_total = 0
        plot_loss_total = 0
        epoch_mean_loss = 0
        brgin_epoch_time = time.time()
        ntokens = len(self.dictionary)
        
        if(direction =='forward'):
            model = self.forward_model
            optimizer = self.forward_optimizer
            lr =  self.forward_lr
        if(direction =='backward'):
            model = self.backward_model
            optimizer = self.backward_optimizer
            lr =  self.backward_lr

        model.train()
        print ('Training Starts ('+direction+') !! Epoch: ', epoch , '\n', '=='*39)

        for batch, i in enumerate(range(0, len(train_data_trimed) , args.batch_size)):
            data, targets = util.get_minibatch(train_data_trimed, train_label_trimed, i, args.batch_size, self.dictionary.padding_id, direction)
            #mask = targets.ne(self.dictionary.padding_id).data
            #print (targets, mask)
            #print (data.size(), data, targets.size(), targets)
            #print (' batch train data; ', len(train_data_trimed) ,'this batch: ', data.size(), ' target size: ', targets.size()) #data size 35 x 20 (here bptt x batch_size) {in gen: batch_size x seq_len}
            #continue
            #data = data.t().contiguous() # after permute data size 20 x 35 (here batch_size x bptt) {in gen: seq_len x batch_size so no need in gen mode }
            #targets = targets.t().contiguous() # same as data
            #if i == 0: print ('train data; AFTER PERMUTE ', train_data.size() ,'this batch: ', data.size(), ' target size: ', targets.size())
            #targets = targets.view(-1)
            #if i == 0: print (' btch first train data; ', train_data.size() ,'this batch: ', data.size(), ' target size: ', targets.size(), targets) #data size 35 x 20 (here bptt x batch_size) {in gen: batch_size x seq_len}
            
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            
            hidden = model.init_hidden(args.batch_size) #for each sentence need to initialize
            hidden = util.repackage_hidden(hidden, args.cuda)
            #print 
            optimizer.zero_grad()
            output, hidden = model(data, hidden)
            #if i == 0: print ('final output: ', output.size(), output,  '\n output.view(-1, ntokens): ', output.view(-1, ntokens))
            loss =  self.criterion(output.view(-1, ntokens), targets)


            # Important if we are using nn.DataParallel()
            #print(loss.data)
            #mean_loss = torch.mean(torch.masked_select(loss.data, mask))
            #assert np.count_nonzero(mask.numpy())*mean_loss.data[0] == loss.data[0] 
            loss.backward()
            optimizer.step()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            clip_grad_norm(model.parameters(), args.clip)
            


            batch_loss += loss.data
            plot_loss_total += loss.data
            epoch_mean_loss += loss.data

            if batch % args.print_every == 0 and batch > 0:
                cur_loss = batch_loss[0] / args.print_every
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:2.2f} | ppl {:4.2f}'.format(
                    epoch, batch, len(train_data_trimed) // args.batch_size, lr,
                                  elapsed * 1000 / args.print_every, cur_loss, math.exp(cur_loss)))
                batch_loss = 0
                start_time = time.time()


            if batch % args.plot_every == 0 and batch > 0:
                plot_loss_avg = plot_loss_total / args.plot_every
                plot_losses.append(plot_loss_avg[0])
                plot_loss_total = 0
            #exit()
        batch +=1 #starts counting from 0
        #print("num of batches: ", batch, ' i: ', i)
        ppl = torch.exp(epoch_mean_loss/batch)[0]
        #print("counter...size of training data: {}".format(counter))
        print('Training epoch: {}  avg loss: {:.2f}  ppl: {:.2f}'.format(epoch, (epoch_mean_loss/batch)[0], ppl) )
        print("Time to complete epoch: {:.2f}".format( time.time() - brgin_epoch_time))
        trainF.write('{}, {}, {}\n'.format(epoch, (epoch_mean_loss/batch)[0], ppl))
        trainF.flush()
        #print('returning plot lossses: ', plot_losses)
        #return plot_losses

    def validate(self, valid_data_trimed, valid_label_trimed , forward_model, backward_model,  epoch, is_test = False):
        # Turn on evaluation mode which disables dropout.
        forward_model.eval()
        backward_model.eval()

       
        total_loss = 0
        ntokens = len(self.dictionary)
        eval_batch_size = args.batch_size #// 2
        
        total_mean_loss_f = total_mean_loss_b = total_mean_loss = 0
         
        for batch, i in enumerate(range(0, len(valid_data_trimed), eval_batch_size)):
            data_f, targets_f = util.get_minibatch(valid_data_trimed, valid_label_trimed, i, eval_batch_size, self.dictionary.padding_id, 'forward', evaluation=True)
            data_b, targets_b = util.get_minibatch(valid_data_trimed, valid_label_trimed, i, eval_batch_size, self.dictionary.padding_id, 'backward', evaluation=True)
            hidden_f = forward_model.init_hidden(eval_batch_size) #for each sentence need to initialize
            hidden_f = util.repackage_hidden(hidden_f, args.cuda)
            hidden_b = backward_model.init_hidden(eval_batch_size) #for each sentence need to initialize
            hidden_b = util.repackage_hidden(hidden_b, args.cuda) 
            output_f, hidden_f = forward_model(data_f, hidden_f)
            output_b, hidden_b = backward_model(data_b, hidden_b)
            output_f = output_f.view(-1, ntokens)
            output_b = output_b.view(-1, ntokens)
            idx = torch.range(output_b.size(0)-1, 0, -1).long()
            idx = torch.autograd.Variable(idx)
            if args.cuda:
                idx = idx.cuda()
            output_b_flipped = output_b.index_select(0, idx)
            assert targets_f.size() == targets_b.size()
            assert output_f.size() == output_b_flipped.size()
            output = output_f + output_b_flipped

            #print('target forward ', targets_f[0], ' backward: ', targets_b[-1] , ' prediction  forward: ', output_f[0][5], ' backward: ', output_b[-1][5], output[0][5] )
            #exit()
            output_flat = output.view(-1, ntokens)
            loss_f = loss =  self.criterion(output_f, targets_f)
            total_mean_loss_f+=loss_f.data
            loss_b = self.criterion(output_b, targets_b)
            total_mean_loss_b+=loss_b.data
            loss =  self.criterion(output_flat, targets_f)
            mean_loss = loss #torch.mean(torch.masked_select(loss.data, mask))
            total_mean_loss += mean_loss.data
        

        # Turn on training mode at the end of validation.
        batch+=1 #starts counting from 0
        forward_model.train()
        backward_model.train()
        avg_loss = total_mean_loss[0]/batch
        ppl = math.exp(avg_loss)
        if not is_test:
            print('Validation epoch: ', epoch)
        else:
             print ("Testing")
        print(" avg loss forward: {:.2f} backward: {:.2f} bidirectional {:.2f} bidir_ppl {:.2f} ".format( total_mean_loss_f[0], total_mean_loss_b[0],  avg_loss,  ppl))
        return avg_loss

        #return avg_loss / num_batches