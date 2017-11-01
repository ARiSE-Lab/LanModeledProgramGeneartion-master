
import os



def test_data_preprocess(path, ):
        train_data,  train_type_data  = tokenize(os.path.join(path, 'all_train.data'), os.path.join(path, 'all_train_type.data'))
        test_data, test_type_data = tokenize(os.path.join(path, 'all_test.data'), os.path.join(path, 'all_test_type.data'))
        return train_data,  train_type_data, test_data, test_type_data

def tokenize(data_path, data_type_path, is_test = False):
        assert os.path.exists(data_path)
        assert os.path.exists(data_type_path)
        # Add words to the dictionary
        new_f = []
        new_tf = []
        with open(data_path, 'r') as df, open(data_type_path, 'r') as tf:
            d_f = df.readlines()
            t_f = tf.readlines()

            for i in range(len(d_f)):
                dline = d_f[i]
                tline =  t_f[i]
                if(len(dline.split()) == len(tline.split()) ):
                    new_f.append(dline)
                    new_tf.append(tline)
        return new_f, new_tf

                
train_data,  train_type_data, test_data, test_type_data = test_data_preprocess('./soft_data/')
print(test_data[50], test_type_data[50], len(train_data), len(train_data),  len(train_type_data), len(test_data), len(test_type_data ))
with open('./soft_data/new_train.data', 'w') as trf:
    for line in train_data:
        trf.write(line)
with open('./soft_data/new_test.data', 'w') as trf:
    for line in test_data:
        trf.write(line)
with open('./soft_data/new_train_type.data', 'w') as trf:
    for line in train_type_data:
        trf.write(line)
with open('./soft_data/new_test_type.data', 'w') as trf:
    for line in test_type_data:
        trf.write(line)
        
