# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:37:46 2020

@author: JRodr
"""

# pip install mlxtend == 0.17.0

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout
from matplotlib import pyplot as plt 
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

#------------------------*opening the signals files*------------------------------------------------------
Data = np.load('ReadyCorrelateSignals.npz') 

X_Test = Data['EvaluateSignals']
X_Test=X_Test[0:458,500:1500,0:22,0:1]/100
Y_Test=Data['EvaluateCodeLabels']

X_Train = Data['TrainSignals']
X_Train=X_Train[0:1823,500:1500,0:22,0:1]/100
Y_Train=Data['TrainCodeLabels']

#------------------*Definition of Convolutional Layers Generator function*----------------------------------------------
def create_new_conv_layer(model, no_filters, filter_size, padding, activation, pool_size, pool_stride ):
    
    model.add(Conv2D(filters= no_filters, kernel_size = filter_size, padding = padding, activation = activation))
    model.add(MaxPool2D(pool_size = pool_size, strides = pool_stride, padding = 'same'  ))
    # model.add(Dropout(0.5))    
    return 

#--------------------*Definition of CNN model*----------------------------------------------------------------------


with tf.device('/GPU:0'):
    
    model = Sequential()
    
    model.add(Conv2D(filters= 64, kernel_size = (10,1), padding = 'same', 
                 activation = 'relu', input_shape = [1000,22,1]  ))
    model.add(MaxPool2D(pool_size = (2,1), strides = (2,1), padding = 'same' ))
    # model.add(Dropout(0.5))
    
    create_new_conv_layer(model, 32, (10, 1), 'same' , 'relu', (2, 1), (2,1)) # Second layer
    create_new_conv_layer(model, 16, (10, 1), 'same' , 'relu', (2, 1), (2,1)) # Third layer
    create_new_conv_layer(model, 16, (10, 1), 'same' , 'relu', (2, 1), (2,1)) # Fourth layer
    create_new_conv_layer(model, 8 , (10, 1), 'same' , 'relu', (2, 1), (2,1)) # Fifth layer
    create_new_conv_layer(model, 64, (1 , 5), 'valid', 'relu', (1, 1), (1,1)) # Sixth layer
    create_new_conv_layer(model, 64, (3 , 1), 'same' , 'relu', (2, 1), (2,1)) # Seventh layer
   
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(units = 1024, activation = 'relu' ))
    # model.add(Dropout(0.5))
    model.add(Dense(units = 4, activation = 'softmax' ))
    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'] )


    history = model.fit(X_Train, Y_Train, batch_size = 50, epochs = 25, verbose = 1, validation_data= (X_Test, Y_Test))
    

plt.plot(range(1,26), history.history['categorical_accuracy'])
plt.plot(range(1,26), history.history['val_categorical_accuracy'])
plt.title('Model Acucracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Val'], loc = 'upper left')
plt.show

plt.plot(range(1,26), history.history['loss'])
plt.plot(range(1,26), history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Val'], loc = 'upper left')
plt.show

Y_Predict = model.predict_classes(X_Test)

mat = confusion_matrix (Y_Test, Y_Predict)
plot_confusion_matrix(mat, figsize = (5,5) )




# #-----------------------*Training and evaluating session start*------------------------------------------------
# init_op = tf.global_variables_initializer()
# saver = tf.train.Saver()

# with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as sess:
#     sess.run(init_op)
#     total_batch = int(1823 / batch_size)
#     print("total batches: ",total_batch)
    
#     epoch_rate_down=0
#     best_kappa=0
#     best_conf_mat=np.zeros([4,4])
#     best_accu=0
#     best_epoch=0

# #-------------------------------------*Training*---------------------------------------------------------    
#     for epoch in range(epochs):
#         avg_cost = 0
#         test_acc_avg=0
#         Nb=0
#         for i in range(total_batch):
#             batch_x, batch_y = Generate_Batch(batch_size,Data)
#             _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
#             avg_cost += c / total_batch
#             Nb+=1
#             if Nb==10:
#                 print("Epoch: ",(epoch+1)," Batch: ", (i+1), " of ", total_batch)
#                 Nb=0
                
# #--------------------------------------*Evaluation*---------------------------------------------------------
#         for j in  range (18):
#             X_Test_batch=X_Test[25*j:25*(j+1) ,0:1000,0:22,0:1]
#             Y_Test_batch=Y_Test[25*j:25*(j+1),0:4]
#             test_acc, predict = sess.run([accuracy,y_],feed_dict={x: X_Test_batch, y: Y_Test_batch})
#             test_acc_avg +=(test_acc*25)/458
#             prediction1[25*j:25*(j+1),0:4]=predict
            
#         X_Test_batch=X_Test[450:458,0:1000,0:22,0:1]
#         Y_Test_batch=Y_Test[450:458,0:4]
#         test_acc, predict = sess.run([accuracy,y_],feed_dict={x: X_Test_batch, y: Y_Test_batch})
#         test_acc_avg +=((test_acc*7)/458)
#         prediction1[450:458,0:4]=predict
               
# #---------------------------------------*Kappa computation*-------------------------------------------------        
#         prediction1=np.around(prediction1)
#         prediction2=np.dot(prediction1,mask)
#         conf_mat=np.zeros([4,4])
#         h=0
#         for h in range (458):
#             a=int(prediction2[h])
#             b=int(reality[h])
#             conf_mat[a][b]= conf_mat[a][b]+1
#         Pcol=np.sum(conf_mat,0)
#         Prow=np.sum(conf_mat,1)
#         Pdia=conf_mat[0][0]+conf_mat[1][1]+conf_mat[2][2]+conf_mat[3][3]
#         Pe=np.dot(Pcol,Prow)/209764 
#         Po=Pdia/458
#         kappa=(Po-Pe)/(1-Pe)
# #-------------------------*printing epoch results and saving best epoch results----------------------------------------------------------------
#         print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "test accuracy: {:.3f}".format(test_acc_avg*100)," Kappa: {:.3f}".format(kappa*100))
#         Accu[epoch]=test_acc_avg*100
#         plt.plot(Accu[0:epoch])
#         if kappa>best_kappa:
#             best_kappa=kappa
#             best_conf_mat=conf_mat
#             best_epoch=epoch
#             best_accu=Accu[epoch]
#     print("\nTraining complete!")
#     plt.plot(Accu[0:epoch-1])