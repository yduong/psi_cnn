import scipy.io as sio
import scipy as scipy
import numpy as np
import os
from concise.preprocessing import encodeDNA
from keras import backend as K
import matplotlib.pyplot as plt


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name
        
def partition_data( number_of_partitions, total_length):
    n = np.array([i for i in range(total_length)])
    np.random.shuffle(n)
    partition = []
    for i in range(number_of_partitions):
        partition.append(n[int(i*total_length/number_of_partitions):int((i+1)*total_length/number_of_partitions)])
    return partition

# get data return train and test data based on cross validation 
def get_data(data_file, partition_file,fold=0):
    data = sio.loadmat(data_file)
    X1  = data['X1']
    X2  = data['X2']
    X1 = np.array(X1,dtype=object)
    X2 = np.array(X2,dtype=object)
    Y = scipy.matrix(data['Y'])
    test_inds = np.load(partition_file)[fold]#partition_data( 10, len(Y))
    encodeSeq1 = data['encodeSeq1']
    encodeSeq2 = data['encodeSeq2']
    
    training_inds = [i for i in range(len(Y)) if i not in test_inds]
    return ({'X1': X1[training_inds], 'X2': X2[training_inds],'seq1': encodeSeq1[training_inds], 'seq2': encodeSeq2[training_inds]}, Y[training_inds]),\
            ({'X1': X1[test_inds], 'X2': X2[test_inds],'seq1': encodeSeq1[test_inds], 'seq2': encodeSeq2[test_inds]}, Y[test_inds])
    
def evaluate_r2_score(y_pred, y_true):
    for i in range(81):
        print(str(i) + " " + str(scipy.stats.pearsonr( y_pred[:,i], np.array(y_true[:,i]).ravel())[0]**2)) 
    
    #print out result
    print("SD New " + str(scipy.stats.pearsonr(np.array(np.sum(y_true[:,49:76], axis = 1)
                                 +np.sum(y_true[:,8:36], axis = 1)).ravel(),
                         np.sum(y_pred[:,49:76], axis = 1)
                         +np.sum(y_pred[:,8:36], axis = 1))[0]**2 ))
    print("SD0 " + str(scipy.stats.pearsonr( y_pred[:,0], np.array(y_true[:,0]).ravel())[0]**2))
    print("SD1 " + str(scipy.stats.pearsonr( y_pred[:,1], np.array(y_true[:,1]).ravel())[0]**2))
    print("SD0CRYPT " + str(scipy.stats.pearsonr( y_pred[:,79], np.array(y_true[:,79]).ravel())[0]**2))
    
    #plot r2 score
    r2score = [(scipy.stats.pearsonr( y_pred[:,i], np.array(y_true[:,i]).ravel())[0]**2) for i in range(81)]
    plt.plot(r2score)
    plt.show()
    
    #plot the best position
    #ax = plt.subplot(221)
    plt.scatter(y_pred[:,55].ravel(),np.array(y_true[:,55]).ravel(), alpha=0.5)
    plt.plot([0,0.6], [0,0.6])
    plt.title("scatter for position 55 " + str(r2score[55]))
    plt.show()
    
    #plot the worse position
    #ax = plt.subplot(222)
    plt.scatter(y_pred[:,10].ravel(),np.array(y_true[:,10]).ravel(), alpha=0.5)
    plt.plot([0,0.6], [0,0.6])
    plt.title("scatter for position 10 " + str(r2score[10]))
    plt.show()
    
    #ax = plt.subplot(223)
    plt.scatter(y_pred[:,31].ravel(),np.array(y_true[:,31]).ravel(), alpha=0.5)
    plt.plot([0,0.6], [0,0.6])
    plt.title("scatter for position 31 " + str(r2score[31]))
    plt.show()
    
    #plot the worse position
    #ax = plt.subplot(224)
    plt.scatter(y_pred[:,73].ravel(),np.array(y_true[:,73]).ravel(), alpha=0.5)
    plt.plot([0,0.6], [0,0.6])
    plt.title("scatter for position 73 " + str(r2score[73]))    
    plt.show()

# get r2 score between y_true and y_pred
def r2_score_k(y_true, y_pred):  
    total_error = K.sum(K.square(y_true- K.mean(y_true)))
    unexplained_error = K.sum(K.square(y_true - y_pred))
    return 1 - unexplained_error/total_error

def sum_r2_score(y_true, y_pred):  
    sd0_r2 = r2_score_k(y_true[:,0], y_pred[:,0]) 
    sd1_r2 = r2_score_k(y_true[:,1], y_pred[:,1]) 
    sdnew_r2 = r2_score_k(y_true[:,79], y_pred[:,79])
    sdscrypt_r2 = r2_score_k(K.sum(y_true[:,8:36], axis = 1) + K.sum(y_true[:,49:76], axis = 1),
                             K.sum(y_pred[:,8:36], axis = 1) + K.sum(y_true[:,49:76], axis = 1)) 
    return sd0_r2 + sd1_r2 + sdnew_r2 + sdscrypt_r2

def get_model(train_data,             
              filters=64, # use powers of 2 - (32, 64, 128)
              motif_width=11, # use odd numbers - 11 or 15
              use_1by1_conv=False, # hp.choice(True, False(
              # regularization
              l1=1e-08,   
              task_specific={'l1_1': 1e-10, 
                      'l1_2' : 1e-09,
                      'l1_3' :1e-08, 
                      },
              # dense layers
              hidden={'n_hidden':1,
                      'n_units': 128,
                      'dropout': 0,
                      'activation': 'relu',
                      'l1': 0
                      },

              activation="sigmoid",# relu
              first_layer_units = 128, # (64, 128, 256, 512)
              lr=0.001,
               loss_fn = "kullback_leibler_divergence"): # (0.01 - 0.0001, log-range)
    # specify the input shape
    input_dna1 = cl.InputDNA(train_data[0]['seq1'].shape[1], name='seq1')

    # seq1
    x1 = cl.ConvDNA(filters=filters, 
                   kernel_size=motif_width, activation="relu")(input_dna1) 
    if use_1by1_conv:
        x1 = kl.Conv1D(filters=filters,
                       kernel_size=1, activation='relu')(x1)
    x1 = kl.Flatten()(x1)  
    
    # seq2
    input_dna2 = cl.InputDNA(train_data[0]['seq2'].shape[1], name='seq2')
    x2 = cl.ConvDNA(filters=filters, 
                    kernel_size=motif_width, activation="relu")(input_dna2)
    if use_1by1_conv:
        x2 = kl.Conv1D(filters=filters,
                       kernel_size=1, activation='relu')(x2)
    x2 = kl.Flatten()(x2)
    x = kl.Concatenate()([x1,x2])
    
    if hidden is not None:
        for i in range(hidden.get('n_hidden')):
            x = kl.Dense(units=hidden.get('n_units'),
                          activation=hidden.get('activation'), 
                          kernel_regularizer=regularizers.l1(hidden.get('l1')))(x) 
            if hidden.get('dropout') != 0:
                x = kl.Dropout(hidden.get('dropout'))(x)
        
    if task_specific is not None:
        # task specific regularization
        x_out1 = kl.Dense(units=2,
                     kernel_regularizer=regularizers.l1(task_specific.get('l1_1')))(x)
        
        # task specific regularization
        x_out2 = kl.Dense(units=int((train_data[1].shape[1] -2)/2),
                     kernel_regularizer=regularizers.l1(task_specific.get('l1_2')))(x)
        
        x_out3 = kl.Dense(units=train_data[1].shape[1] -2 - int((train_data[1].shape[1] -2)/2),
                     kernel_regularizer=regularizers.l1(task_specific.get('l1_3')))(x)
        x = kl.Concatenate()([x_out1, x_out2, x_out3])
        x = kl.Activation('softmax')(x)
    else:
        x = kl.Dense(units=train_data[1].shape[1],
                     activation="softmax",
                     kernel_regularizer=regularizers.l1(l1))(x)
    
    model = Model(inputs=[input_dna1,input_dna2], outputs=x)

    model.compile(optimizer=ko.Adam(lr=lr), loss=loss_fn, metrics=[r2_score_k])
    return model