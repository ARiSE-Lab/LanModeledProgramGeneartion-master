###############################################################################
# Author: Md Rizwan Parvez
# Project: LanModeledProgramGeneration
# Date Created: 4/1/2017
# Some codes are from Wasi Ahmad main.py
# File Description: This is the main script from where all experimental
# execution begins.
###############################################################################
import torch.nn as nn
import util, data #, helper, train
import torch, random
from torch import optim
import model_rnd
import time, math, os
from argparse import ArgumentParser
from embedding_layer import Embedding_Drop_Layer
from torch.autograd import Variable
import train


args = util.get_args()


print(args)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('='*90, "\nWARNING: You have a CUDA device, so you should probably run with --cuda\n", '='*89)
    else:
    	#torch.cuda.set_device(2)
    	torch.cuda.manual_seed(args.seed)


###############################################################################
# Load var data
###############################################################################

#### fix this
# corpus = data.Corpus(args.data)
corpus = data.Corpus(args)
print('Train set size = ', len(corpus.train_data), len(corpus.train_label))
print('Test set size = ', len(corpus.test_data), len(corpus.test_label))
print('Vocabulary size = ', len(corpus.dictionary))


train_var_data_trimed, train_var_label_trimed = util.batchify(corpus.train_data, corpus.train_label, args.batch_size, args.cuda) #[82915, 20] batch size = 20, it's kinda seq len in gen sense
valid_var_data_trimed, valid_var_label_trimed = util.batchify(corpus.valid_data, corpus.valid_label, args.batch_size, args.cuda) #[82915, 20] batch size = 20, it's kinda seq len in gen sense 
test_var_data_trimed, test_var_label_trimed = util.batchify(corpus.test_data, corpus.test_label, args.batch_size, args.cuda) #[82915, 20] batch size = 20, it's kinda seq len in gen sense


###############################################################################
# Load type data
###############################################################################


args.train_data = args.train_data.rstrip('.data')+'_type.data'
args.valid_data = args.valid_data.rstrip('test.data')+'test_type.data'
args.test_data = args.test_data.rstrip('test.data')+'test_type.data'

corpus_type = data.Corpus(args)
print('Train set size = ', len(corpus_type.train_data), len(corpus_type.train_label))
print('Test set size = ', len(corpus_type.test_data), len(corpus_type.test_label))
print('Vocabulary size = ', len(corpus_type.dictionary))
train_type_data_trimed, train_type_label_trimed = util.batchify(corpus_type.train_data, corpus_type.train_label, args.batch_size, args.cuda) #[82915, 20] batch size = 20, it's kinda seq len in gen sense
valid_type_data_trimed, valid_type_label_trimed = util.batchify(corpus_type.valid_data, corpus_type.valid_label, args.batch_size, args.cuda) #[82915, 20] batch size = 20, it's kinda seq len in gen sense 
test_type_data_trimed, test_type_label_trimed = util.batchify(corpus_type.test_data, corpus_type.test_label, args.batch_size, args.cuda) #[82915, 20] batch size = 20, it's kinda seq len in gen sense


if args.debug:

	# train_data_trimed = train_data_trimed[:50]
    # valid_data_trimed = valid_data_trimed[:50]
    # test_data_trimed = test_data_trimed[:50]

    # train_label_trimed = train_label_trimed[:50]
    # valid_label_trimed = valid_label_trimed[:50]
    # test_label_trimed = test_label_trimed[:50]
    test_var_data_trimed = test_var_data_trimed[:50]
    test_var_label_trimed = test_var_label_trimed[:50]
    test_type_data_trimed = test_type_data_trimed[:50]
    test_type_label_trimed = test_type_label_trimed[:50]

assert len(train_type_data_trimed) == len(train_type_label_trimed)
assert len(valid_type_data_trimed) == len(valid_type_label_trimed)
assert len(test_type_data_trimed) == len(test_type_label_trimed)

assert len(train_var_data_trimed) == len(train_var_label_trimed)
assert len(valid_var_data_trimed) == len(valid_var_label_trimed)
assert len(test_var_data_trimed) == len(test_var_label_trimed)

# assert len(train_var_data_trimed) == len(train_type_label_trimed)
# assert len(valid_var_data_trimed) == len(valid_type_label_trimed)
assert len(test_var_data_trimed) == len(test_type_label_trimed)



# print (batchify([2,3,4,3,4,355,4,342,90], 2))
print('train_batches: size: ', len(train_var_data_trimed) ) #, 'seq len: ', len(train_data_trimed[0]), '1st instance: ', train_data_trimed[0][:50], '1st label: ', train_label_trimed[0][:50] )# , train_batches[0][0].sentence1)


# # ###############################################################################
# # # Build the model
# # ###############################################################################

model_f = model_rnd.LanguageModel(corpus.dictionary, args)
model_b = model_rnd.LanguageModel(corpus.dictionary, args)

if args.cuda:
    torch.cuda.set_device(args.gpu)
    model_f.cuda()
    model_b.cuda()



criterion = nn.CrossEntropyLoss(size_average=True)

# ###############################################################################
# # testing code
# ###############################################################################


f_f = 'forward_model_best.pth.tar'
b_f = 'backward_model_best.pth.tar'

if os.path.isfile(os.path.join(args.log_dir, f_f)) and os.path.isfile(os.path.join(args.log_dir, b_f)):
    print("=> Starting loading  best models for testing: from ", os.path.join(args.log_dir, f_f), ' and ', os.path.join(args.log_dir, b_f) )
    checkpoint_forward= torch.load(os.path.join(args.log_dir, f_f))
    checkpoint_backward= torch.load(os.path.join(args.log_dir, b_f))
    args.start_epoch = checkpoint_forward['epoch']
    best_perplexity_forward = checkpoint_forward['perplexity']
    best_perplexity_backward = checkpoint_backward['perplexity']
    model_f.load_state_dict(checkpoint_forward['state_dict'])
    model_b.load_state_dict(checkpoint_backward['state_dict'])
    print("=> Finished loading  best models for testing")



model_tf = model_rnd.LanguageModel(corpus_type.dictionary, args)
model_tb = model_rnd.LanguageModel(corpus_type.dictionary, args)
if args.cuda:
    torch.cuda.set_device(args.gpu)
    model_tf.cuda()
    model_tb.cuda()
f_f = 'forward_model_best.pth.tar'
b_f = 'backward_model_best.pth.tar'

if os.path.isfile(os.path.join(args.log_type_dir, f_f)) and os.path.isfile(os.path.join(args.log_type_dir, b_f)):
    print("=> Starting loading  best models for testing: from ", os.path.join(args.log_type_dir, f_f), ' and ', os.path.join(args.log_type_dir, b_f) )
    checkpoint_forward= torch.load(os.path.join(args.log_type_dir, f_f))
    checkpoint_backward= torch.load(os.path.join(args.log_type_dir, b_f))
    args.start_epoch = checkpoint_forward['epoch']
    best_perplexity_forward_type= checkpoint_forward['perplexity']
    best_perplexity_backward_type = checkpoint_backward['perplexity']
    model_tf.load_state_dict(checkpoint_forward['state_dict'])
    model_tb.load_state_dict(checkpoint_backward['state_dict'])
    print("=> Finished loading  best models for testing")


def get_symbol_table(data, types):
	# print(data, " \n types: ", types)
	data = data.data[0]
	types = types.data[0]
	# print(data, " \n types: ", types)
	id_map ={}
	i = 0
	for i  in range(len(data)):
		pos = data[i]
		tp = types[i]
		id_map.update({pos:tp})
	return id_map



def infer(valid_data_trimed, valid_label_trimed, valid_type_data_trimed, valid_type_label_trimed, forward_model, backward_model,  forward_type_model, backward_type_model, var_dict, type_dict, criterion):
        # Turn on evaluation mode which disables dropout.
        forward_model.eval()
        backward_model.eval()
        forward_type_model.eval()
        backward_type_model.eval()

        total_mean_loss_f = 0
        total_mean_loss_b = 0
        total_mean_loss_cb = 0
        total_mean_loss_tf = 0
        total_mean_loss_tb = 0
        total_mean_loss = 0
        total_mean_lossc = 0
        total_mean_losst = 0

        ntokens = len(var_dict)
        type_ntokens = len(type_dict)
        eval_batch_size = args.batch_size #// 2
    
         
        for batch, i in enumerate(range(0, len(valid_data_trimed), eval_batch_size)):
            data_f, targets_f = util.get_minibatch(valid_data_trimed, valid_label_trimed, i, eval_batch_size, var_dict.padding_id, 'forward', evaluation=True)
            data_b, targets_b = util.get_minibatch(valid_data_trimed, valid_label_trimed, i, eval_batch_size, var_dict.padding_id, 'backward', evaluation=True)



            data_tf, targets_tf = util.get_minibatch(valid_type_data_trimed, valid_type_label_trimed, i, eval_batch_size, type_dict.padding_id, 'forward', evaluation=True)
            data_tb, targets_tb = util.get_minibatch(valid_type_data_trimed, valid_type_label_trimed, i, eval_batch_size, type_dict.padding_id, 'backward', evaluation=True)



            #mask = data.ne(dictionary.padding_id)
            hidden_f = forward_model.init_hidden(eval_batch_size) #for each sentence need to initialize
            hidden_b = backward_model.init_hidden(eval_batch_size) #for each sentence need to initialize

            hidden_f = util.repackage_hidden(hidden_f, args.cuda)
            hidden_b = util.repackage_hidden(hidden_b, args.cuda)


            hidden_tf = forward_type_model.init_hidden(eval_batch_size) #for each sentence need to initialize
            hidden_tb = backward_type_model.init_hidden(eval_batch_size) #for each sentence need to initialize

            hidden_tf = util.repackage_hidden(hidden_tf, args.cuda)
            hidden_tb = util.repackage_hidden(hidden_tb, args.cuda)



            output_f, hidden_f = forward_model(data_f, hidden_f)
            output_b, hidden_b = backward_model(data_b, hidden_b)

            output_tf, hidden_tf = forward_type_model(data_tf, hidden_tf)
            output_tb, hidden_tb = backward_type_model(data_tb, hidden_tb)



            output_flat_f = output_f.view(-1, ntokens) # (batch x seq) x ntokens 
            output_flat_b = output_b.view(-1, ntokens)


            output_flat_tf = output_tf.view(-1, type_ntokens) # (batch x seq) x ntokens 
            output_flat_tb = output_tb.view(-1, type_ntokens)



            output_flat_f_t = output_flat_f
            output_flat_b_t = output_flat_b

            output_flat_tf_t = output_flat_tf
            output_flat_tb_t = output_flat_tb


            m = nn.Softmax()
            output_flat_f = m(output_flat_f)
            output_flat_b = m(output_flat_b)


            output_flat_tf = m(output_flat_tf)
            output_flat_tb = m(output_flat_tb)



            x = 0.5
            idx = torch.range(output_flat_b.size(0)-1, 0, -1).long()
            idx = torch.autograd.Variable(idx)
            if args.cuda:
                idx = idx.cuda()
            output_flat_b_flipped = output_flat_b.index_select(0, idx)
            assert targets_f.size() == targets_b.size()
            assert output_flat_f.size() == output_flat_b_flipped.size()
            output = x*output_flat_f + (1-x)*output_flat_b_flipped
            output_flat = output.view(-1, ntokens) 



            tidx = torch.range(output_flat_tb.size(0)-1, 0, -1).long()
            tidx = torch.autograd.Variable(tidx)
            if args.cuda:
                tidx = tidx.cuda()
            output_flat_tb_flipped = output_flat_tb.index_select(0, tidx)
            assert targets_tf.size() == targets_tb.size()
            assert output_flat_tf.size() == output_flat_tb_flipped.size()
            outputt = x*output_flat_tf + (1-x)*output_flat_tb_flipped
            output_flatt = outputt.view(-1, type_ntokens) 

            # if(i==0): 
                # util.view_bidirection_calculation(output_flat_f, output_flat_b_flipped, output_flat, targets_f, self.dictionary, k = 5)

            numwords = output_flat_b.size()[0]
            symbol_table = get_symbol_table(data_b, data_tb)
            # print(symbol_table)
            output_flat_cb= output_flat_b_t.clone()#torch.FloatTensor(output_flat_b.size()).zero_()#copy[:]
            # print(output_flat_cb[0])
           	# make_symbol_table()
  
            for idxx in range(numwords): #for each of words at prediction time
            	# print (" \n\n-----actual target word: ", var_dict.idx2word[targets_b.data[idxx]], '------')
            	for pos in set(data_b.data[0]): #that means if the token is in the method or unknown for the method
            		tp = symbol_table[pos]
            		# print (" \n\n----- prediction word: ", var_dict.idx2word[pos], ' type: ', type_dict.idx2word[tp]),'------'
            		# print (" backward prob of type ", output_flat_tb_t.data[idxx][tp], "\n before chnaging var back prob was: ", output_flat_cb.data[idxx][pos], ' ori: ', output_flat_b_t.data[idxx][pos])
            		output_flat_cb.data[idxx][pos] += output_flat_tb_t.data[idxx][tp]
            		# print(" var backward prob ", output_flat_b_t[idxx][pos], ' after softmax ', output_flat_b[idxx][pos] ,' now it is: ', output_flat_cb.data[idxx][pos])
            	# exit()
            # 		print(' target type : ', targets_tb[0])
            # 	exit()
            		# output_flat_cb.data[idxx][pos] +=  output_flat_b_t.data[idxx][pos]
            # exit()	
            # print('output_flat_b[0]: ', output_flat_b[0][8])
            # print('output_flat_cb[0]: ', output_flat_cb[0][8])
            # exit()
            # output_flat_cb = Variable(output_flat_cb)
            # if(args.cuda): output_flat_cb = output_flat_cb.cuda()


            loss_f = criterion(output_flat_f_t, targets_f)
            loss_b = criterion(output_flat_b_t, targets_b)



            loss_tf = criterion(output_flat_tf_t, targets_tf)
            loss_tb = criterion(output_flat_tb_t, targets_tb)


            loss = nn.functional.nll_loss(torch.log(output_flat), targets_f, size_average=True)

            losst = nn.functional.nll_loss(torch.log(output_flatt), targets_tf, size_average=True)

            loss_cb =  nn.functional.nll_loss(torch.log(m(output_flat_cb)), targets_b, size_average=True)

            assert loss_cb[0]>0
            assert loss_cb[0]>loss_b[0]

            # print ('loss_b: ',loss_b.data[0])
            # print ('loss_cb: ',loss_cb.data[0])

            mean_loss_f = loss_f #torch.mean(torch.masked_select(loss.data, mask))
            mean_loss_b = loss_b #torch.mean(torch.masked_select(loss.data, mask))
            mean_loss = loss

            mean_loss_cb = loss_cb


            mean_loss_tf = loss_tf #torch.mean(torch.masked_select(loss.data, mask))
            mean_loss_tb = loss_tb #torch.mean(torch.masked_select(loss.data, mask))
            mean_losst = losst




            total_mean_loss_f += mean_loss_f.data
            total_mean_loss_b += mean_loss_b.data
            total_mean_loss += mean_loss.data

            total_mean_loss_cb +=mean_loss_cb.data

            total_mean_loss_tf += mean_loss_tf.data
            total_mean_loss_tb += mean_loss_tb.data
            total_mean_losst += mean_losst.data


            
            if(batch%500==0): print ("done batch ", batch, ' of ', len(valid_data_trimed)/ eval_batch_size)

        batch +=1 #starts counting from 0 hence total num batch (after finishing) = batch + 1
        forward_model.train()
        backward_model.train()
        forward_type_model.train()
        backward_type_model.train()



        avg_loss_f = total_mean_loss_f[0]/batch
        avg_loss_b = total_mean_loss_b[0]/batch
        avg_loss = total_mean_loss[0]/batch
        avg_loss_cb = total_mean_loss_cb[0]/batch


        avg_loss_tf = total_mean_loss_tf[0]/batch
        avg_loss_tb = total_mean_loss_tb[0]/batch
        avg_losst = total_mean_losst[0]/batch



        ppl_f = math.exp(avg_loss_f)
        ppl_b = math.exp(avg_loss_b)
        ppl_cb = math.exp(avg_loss_cb)
        ppl = math.exp(avg_loss)


        ppl_tf = math.exp(avg_loss_tf)
        ppl_tb = math.exp(avg_loss_tb)
        pplt = math.exp(avg_losst)



             
        print('Var model direction {} avg loss: {:.2f}  ppl: {:.2f} '.format(  'forward', avg_loss_f, ppl_f))
        print('Var model direction {} avg loss: {:.2f}  ppl: {:.2f} '.format(  'backward', avg_loss_b, ppl_b))
        print('Var model direction {} avg loss: {:.2f}  ppl: {:.2f} '.format(  'bidirectional', avg_loss, ppl))

        print('Type model direction {} avg loss: {:.2f}  ppl: {:.2f} '.format(  'combined backward', avg_loss_cb, ppl_cb))

        print('Type model direction {} avg loss: {:.2f}  ppl: {:.2f} '.format(  'forward', avg_loss_tf, ppl_tf))
        print('Type model direction {} avg loss: {:.2f}  ppl: {:.2f} '.format( 'backward', avg_loss_tb, ppl_tb))
        print('Type model direction {} avg loss: {:.2f}  ppl: {:.2f} '.format(  'bidirectional', avg_losst, pplt))




        #combined inference of the backward models as they are the best models in both cases
        # output_flat_tf (batch x seq) x ntokens 






infer(test_var_data_trimed, test_var_label_trimed, test_type_data_trimed, test_type_label_trimed, model_f, model_b, model_tf, model_tb, corpus.dictionary, corpus_type.dictionary, criterion)
     



