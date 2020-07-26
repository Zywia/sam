from __future__ import division
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Dense
from keras.models import Model
import os, cv2, sys
import numpy as np
from config import *
from utilities import preprocess_images, preprocess_maps, preprocess_fixmaps, postprocess_predictions
from models import sam_vgg, sam_resnet, kl_divergence, correlation_coefficient, nss



IMAGENET_PATH = "/media/data2/infotech/datasets/imagenet/"
TRAIN_FOLDER_NAME = "train"
VALIDATION_FOLDER_NAME = "val"

LIST_OF_FOLDERS_SLASH_CLASSES =  np.array( os.listdir(os.path.join(IMAGENET_PATH, TRAIN_FOLDER_NAME)))




def extract_images_and_their_classes(train_or_val, labels_part=0.1):   
    list_of_images_paths = { x: np.array(os.listdir(os.path.join(IMAGENET_PATH, train_or_val, x)))
                        [:int (len(os.listdir(os.path.join(IMAGENET_PATH, train_or_val, x)))
                                * labels_part)]
                        for x in LIST_OF_FOLDERS_SLASH_CLASSES}


    this_is_going_out = np.array( [(z, x)  for x, y 
    in list_of_images_paths.items() for z in y] )

    shake_it = np.random.permutation(this_is_going_out.shape[0])

    return this_is_going_out[shake_it]


def generator(b_s, phase_gen='train'):
    def add_path_information(image_and_class):
        return np.array(os.path.join(IMAGENET_PATH, phase_gen, image_and_class[1], image_and_class[0]), dtype=object)
    
    images_and_their_classes = extract_images_and_their_classes(phase_gen).astype(object)
    name_and_id_of_class = { y : x + 1 for x, y in enumerate(LIST_OF_FOLDERS_SLASH_CLASSES)}

    counter = 0
    while True:
        images = np.apply_along_axis(add_path_information, axis=1, 
        arr=images_and_their_classes[counter:counter + b_s])
        y = np.zeros((b_s, 1000))
        indicies = np.array([name_and_id_of_class[x] for x in images_and_their_classes[counter:counter + b_s, 1]])
        
        y[:, indicies - 1] = 1
        # y = enc.transform(images_and_their_classes[counter:counter + b_s, 1].reshape(-1, 1)).toarray()
        
        yield preprocess_images(images, shape_r, shape_c), y
        counter = (counter + b_s) % images_and_their_classes.shape[0]


def generator_test(b_s, imgs_test_path):
    images = [imgs_test_path + f for f in os.listdir(imgs_test_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()

    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    counter = 0
    while True:
        yield [preprocess_images(images[counter:counter + b_s], shape_r, shape_c), gaussian]
        counter = (counter + b_s) % len(images)

if __name__ == '__main__':
    gen = generator(b_s=32)
    im, label =  next(gen)
    print( im.shape) 
    
    # im, label =  next(gen)
    # print( im.shape) 

    if len(sys.argv) == 1:
        raise NotImplementedError
    else:
        phase = sys.argv[1]
        x = Input((3, shape_r, shape_c))
        x_maps = Input((nb_gaussian, shape_r_gt, shape_c_gt))

        based = sam_resnet([x, x_maps])
        

        m = Model(input=[x, x_maps], output=sam_resnet([x, x_maps]))
        
        m.load_weights('weights/sam-resnet_salicon_weights.pkl')


        # Disassemble layers
        layers = [l for l in m.layers]

        output = GlobalAveragePooling2D()(layers[177].output)
        output = Dense(1000, activation='softmax')(output)
        new_m = Model(input=x, output=output)
        # print(new_m.summary())

        # print ( [l.name for l in layers])
        for num, l in enumerate(layers):
            if l.name.startswith("attentiveconvlstm"):
                print(num, l.name)
            # print ( num, l.name)
        




        print("Compiling SAM-ResNet")
        # print(m.summary())
        # m.compile(RMSprop(lr=1e-4), 
        # loss=[kl_divergence, correlation_coefficient, nss])
    
        new_m.compile(optimizer='SGD',
          loss='categorical_crossentropy',
          metrics=['accuracy'])


        print("Training SAM-ResNet")

        new_m.fit_generator(generator(b_s=b_s), nb_imgs_train, nb_epoch=nb_epoch,
                            validation_data=generator(b_s=b_s, phase_gen='val'), nb_val_samples=nb_imgs_val,
                            callbacks=[EarlyStopping(patience=3),
                                        ModelCheckpoint('weights.sam-resnet.{epoch:02d}-{val_loss:.4f}.pkl', save_best_only=True)])


        if phase == 'train':
            if nb_imgs_train % b_s != 0 or nb_imgs_val % b_s != 0:
                print("The number of training and validation images should be a multiple of the batch size. Please change your batch size in config.py accordingly.")
                exit()

          
            print("Training SAM-ResNet")
            m.fit_generator(generator(b_s=b_s), nb_imgs_train, nb_epoch=nb_epoch,
                            validation_data=generator(b_s=b_s, phase_gen='val'), nb_val_samples=nb_imgs_val,
                            callbacks=[EarlyStopping(patience=3),
                                        ModelCheckpoint('weights.sam-resnet.{epoch:02d}-{val_loss:.4f}.pkl', save_best_only=True)])

        # elif phase == "test":
            # Output Folder Path
        #     output_folder = 'predictions/'

        #     if len(sys.argv) < 2:
        #         raise SyntaxError
        #     imgs_test_path = sys.argv[2]

        #     file_names = [f for f in os.listdir(imgs_test_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        #     file_names.sort()
        #     nb_imgs_test = len(file_names)

        #     if nb_imgs_test % b_s != 0:
        #         print("The number of test images should be a multiple of the batch size. Please change your batch size in config.py accordingly.")
        #         exit()

         
        #     print("Loading SAM-ResNet weights")
        #     m.load_weights('weights/sam-resnet_salicon_weights.pkl')

        #     print("Predicting saliency maps for " + imgs_test_path)
        #     predictions = m.predict_generator(generator_test(b_s=b_s, imgs_test_path=imgs_test_path), nb_imgs_test)[0]


        #     for pred, name in zip(predictions, file_names):
        #         original_image = cv2.imread(imgs_test_path + name, 0)
        #         res = postprocess_predictions(pred[0], original_image.shape[0], original_image.shape[1])
        #         cv2.imwrite(output_folder + '%s' % name, res.astype(int))
        # else:
        #     raise NotImplementedError
