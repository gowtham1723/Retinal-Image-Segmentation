###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################


import numpy as np
import configparser

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout,Dense,GlobalAveragePooling2D,Flatten,Add,merge,BatchNormalization,Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
from keras.applications.vgg19 import VGG19,preprocess_input,decode_predictions
import keras
import tensorflow as tf

import sys
sys.path.insert(0, './lib/')
from help_functions import *
K.set_image_data_format('channels_first')
#function to obtain data for training/testing (validation)
from extract_patches import get_data_training

def se_block(in_block, ch, ratio=16):
    x = GlobalAveragePooling2D()(in_block)
    x = Dense(ch//ratio, activation='relu')(x)
    x = Dense(ch, activation='sigmoid')(x)
    return np.multiply()([in_block, x])

def RNNlayer(input_layer,n):
    convx = Conv2D(n, (3, 3), activation='relu', padding='same',data_format='channels_first')(input_layer)
    convb= Conv2D(n, (1,1), activation='relu', padding='same',data_format='channels_first')(input_layer)
    conv_out=Add()([convb, convx])
    conv_out=Activation('relu')(conv_out)
    return conv_out

def TRPlayer(input_layer,n):
    convx = Conv2D(n, (1, 1), activation='relu', padding='same',data_format='channels_first')(input_layer)
    convy = Conv2D(n, (3, 3), activation='relu', padding='same',data_format='channels_first')(input_layer)
    convz = Conv2D(n, (5, 5), activation='relu', padding='same',data_format='channels_first')(input_layer)
    #convl = MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same',data_format='channels_first')(input_layer)
    conv_con=concatenate([convx,convy,convz],axis=1)
    conv_con = Conv2D(n, (3, 3), padding='same',data_format='channels_first')(conv_con)
    conv_con=BatchNormalization(axis=1)(conv_con)
    conv_con=Activation('relu')(conv_con)
    conv_out=Add()([input_layer, conv_con])
    #conv_out = Conv2D(n, (3,3),padding='same',data_format='channels_first')(conv_out)
    #conv_out=Activation('relu')(conv_out)
    #conv_out=Activation('relu')(conv_out)
    return conv_out


def get_unet_trp(n_ch,patch_height,patch_width,pretrained_weights = None):
    inputs = Input((n_ch, patch_height, patch_width))

    
    conv3 = Conv2D(32, (3, 3), padding='same',data_format='channels_first')(inputs)
    conv3=BatchNormalization(axis=1)(conv3)
    conv3=Activation('relu')(conv3)
    conv3 = Conv2D(32, (3, 3), padding='same',data_format='channels_first')(conv3)
    conv3=BatchNormalization(axis=1)(conv3)
    conv3=Activation('relu')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv3 = TRPlayer(conv3,32)
    #conv3=concatenate([conv3t,conv3],axis=1)
    #conv3 = TRPlayer(conv3,32)
    #conv3=RNNlayer(conv3,32)
    #conv3=RNNlayer(conv3,32)
    #conv3=RNNlayer(conv3,32)
    
    #
    conv4 = Conv2D(64, (3, 3), padding='same',data_format='channels_first')(pool2)
    conv4=BatchNormalization(axis=1)(conv4)
    conv4=Activation('relu')(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4) 
    conv4=BatchNormalization(axis=1)(conv4)
    conv4=Activation('relu')(conv4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4 = TRPlayer(conv4,64)
    #conv4=concatenate([conv4t,conv4],axis=1)
    #conv4=RNNlayer(conv4,64)
    #conv4=RNNlayer(conv4,64)
    #
    #
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool3)
    conv5=BatchNormalization(axis=1)(conv5)
    conv5=Activation('relu')(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
    conv5=BatchNormalization(axis=1)(conv5)
    conv5=Activation('relu')(conv5)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv5)
    conv5 = TRPlayer(conv5,128)
    #conv5=concatenate([conv5t,conv5],axis=1)
    #conv5=RNNlayer(conv5,128)

    convx = Conv2D(256, (3, 3), padding='same',data_format='channels_first')(pool4)
    convx=BatchNormalization(axis=1)(convx)
    convx=Activation('relu')(convx)
    convx = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_first')(convx)
    #pool5 = MaxPooling2D(pool_size=(2, 2))(convx)
    #convx = TRPlayer(convx,256)
    #

    #convq = Conv2D(512, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool5)
    #convq = Dropout(0.2)(convq)
    #convq = Conv2D(512, (3, 3), activation='relu', padding='same',data_format='channels_first')(convq)
    
    #up = UpSampling2D(size=(2, 2))(convq)
    #up = Conv2D(256, (2, 2), activation='relu', padding='same',data_format='channels_first')(up)
    #up=concatenate([up,convx],axis=1)
    #convw = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_first')(up)
    #convw = Dropout(0.2)(convw)
    #convw = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_first')(convw)

    up1 = UpSampling2D(size=(2, 2))(convx)
    up1 = Conv2D(128, (2, 2), activation='relu', padding='same',data_format='channels_first')(up1)
    up1=concatenate([up1,conv5],axis=1)
    convy = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    convy = Dropout(0.2)(convy)
    convy = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(convy)

    up2 = UpSampling2D(size=(2, 2))(convy)
    up2 = Conv2D(64, (2, 2), activation='relu', padding='same',data_format='channels_first')(up2)
    up2=concatenate([up2,conv4],axis=1)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv6)
    #
    up3 = UpSampling2D(size=(2, 2))(conv6)
    up3 = Conv2D(32, (2, 2), activation='relu', padding='same',data_format='channels_first')(up3)
    up3=concatenate([up3,conv3],axis=1)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up3)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv7)

    #
    conv10 = Conv2D(2, (1, 1), activation='relu', padding='same',data_format='channels_first')(conv7)
    conv10 = core.Reshape((2,patch_height*patch_width))(conv10)
    conv10 = core.Permute((2,1))(conv10)
    ############
    conv10 = core.Activation('softmax')(conv10)

    model = Model(input=inputs, output=conv10)

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

#========= Load settings from Config file
config = configparser.RawConfigParser()
config2=configparser.RawConfigParser()
config.read('configuration.txt')
config2.read('configuration2.txt')
#patch to the datasets
path_data = config.get('data paths','path_local')
#Experiment name
name_experiment = config.get('experiment name', 'name')
#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))
weights_path = config2.get('data paths', 'weights')


#============ Load the data and divided in patches
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
    DRIVE_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth'),  #masks
    patch_height = int(config.get('data attributes', 'patch_height')),
    patch_width = int(config.get('data attributes', 'patch_width')),
    N_subimgs = int(config.get('training settings', 'N_subimgs')),
    inside_FOV = config.getboolean('training settings', 'inside_FOV') #select the patches only inside the FOV  (default == True)
)



#========= Save a sample of what you're feeding to the neural network ==========
N_sample = min(patches_imgs_train.shape[0],40)
visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_imgs")#.show()
visualize(group_images(patches_masks_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_masks")#.show()


#=========== Construct and save the model arcitecture =====
n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]
model = get_unet_trp(n_ch,patch_height,patch_width)  #the y-net model
print("Check: final output of the network:")
print(model.output_shape)
#plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)



#============  Training ==================================
checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased


# def step_decay(epoch):
#     lrate = 0.01 #the initial learning rate (by default in keras)
#     if epoch==100:
#         return 0.005
#     else:
#         return lrate
#
# lrate_drop = LearningRateScheduler(step_decay)

patches_masks_train = masks_Unet(patches_masks_train)  #reduce memory consumption
model.fit([patches_imgs_train], [patches_masks_train], epochs=N_epochs, batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.1, callbacks=[checkpointer])


#========== Save and test the last model ===================
model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)
#test the model
# score = model.evaluate(patches_imgs_test, masks_Unet(patches_masks_test), verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])


















#
