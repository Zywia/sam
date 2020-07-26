import os
import numpy as np



IMAGENET_PATH = "/media/data2/infotech/datasets/imagenet/"
TRAIN_FOLDER_NAME = "train"
VALIDATION_FOLDER_NAME = "val"

LIST_OF_FOLDERS_SLASH_CLASSES = os.listdir(os.path.join(IMAGENET_PATH, TRAIN_FOLDER_NAME))

phase = "train"
def add_path_information(image_and_class):
    # print (os.path.join(IMAGENET_PATH, phase, image_and_class[1], image_and_class[0]))
    return np.array(os.path.join(IMAGENET_PATH, phase, image_and_class[1], image_and_class[0]), dtype=object)

    
def extract_images_and_their_classes(train_or_val):   
    list_of_images_paths = { x: os.listdir(os.path.join(IMAGENET_PATH, train_or_val, x))
    for x in LIST_OF_FOLDERS_SLASH_CLASSES}

    this_is_going_out = np.array( [(z, x)  for x, y 
    in list_of_images_paths.items() for z in y] )

    shake_it = np.random.permutation(this_is_going_out.shape[0])

    return this_is_going_out[shake_it]


# list_of_images_paths = { x: os.listdir(os.path.join(IMAGENET_PATH, TRAIN_FOLDER_NAME, x))
#  for x in LIST_OF_FOLDERS_SLASH_CLASSES}


# enc = preprocessing.OneHotEncoder()
# LIST_OF_FOLDERS_SLASH_CLASSES = np.array(LIST_OF_FOLDERS_SLASH_CLASSES)
# enc.fit(LIST_OF_FOLDERS_SLASH_CLASSES.reshape(-1,1))
# one_hot = enc.transform(LIST_OF_FOLDERS_SLASH_CLASSES[0:2].reshape(-1, 1)).toarray() 
# list_of_images_names_and_their_class = [(z, x)  for x, y in list_of_images_paths.items() for z in y]
# print(np.array(extract_images_and_their_classes("train"))[0] )
im = extract_images_and_their_classes("train")
# print ( os.path.join(im[0:2, 0],  im[0:2, 1])) 
print (np.apply_along_axis(add_path_information, axis=1, arr=im[:10]))

name_and_id_of_class = { y : x + 1 for x, y in enumerate(LIST_OF_FOLDERS_SLASH_CLASSES)}
test = ["n02981792", "n02981792"]
test1 = [name_and_id_of_class[x] for x in test]

batch_size = 2
y = np.zeros((batch_size, 1000, 1))
y[:, np.array(test1) - 1] = 1 
print(y.shape)
print(np.where(y))

print (test1)
# print (name_and_id_of_class["n02981792", "n02981792"] )
print (len(extract_images_and_their_classes("val")))
classes = np.array( list(range(20)))


# print (name_and_id_of_class)
print (np.zeros((1000, 1)).shape)


# print (len(list_of_images_names_and_their_class) )
# print(np.argwhere(one_hot) )
