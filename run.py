from bot_server.Updater import Updater
from mtcnn.mtcnn import MTCNN
from PIL import Image
from matplotlib.pyplot import imread
import tempfile
import skimage
import sklearn
import numpy as np
import os
import pickle
import tensorflow as tf

target_shape=(200,200)
detector = MTCNN()
with open("./src/CNN_training/Classifier/svm2.sav", "rb") as handle:
        classifier = pickle.load(handle)
embedding = tf.keras.models.load_model("./src/CNN_training/siamese_embedding_model")

# BotFather Token to exploit created Telegram bot
TOKEN = "5758831231:AAFqcnYeS79nJ19ZwIoyJWbVv6Cbn6jSbnI"


def find_faces(image_path):
    res = []
    img = imread(image_path)
    result_list = detector.detect_faces(img)
    if len(result_list) == 1:
        [X, Y, W, H] = result_list[0]['box']
        crop = img[Y:Y+H, X:X+W]
        faces_found = detector.detect_faces(crop)
        if len(faces_found) == 1:
            res.append(crop)
    elif len(result_list) > 1:
        for result in result_list:
            [X, Y, W, H] = result['box']
            crop = img[Y:Y+H, X:X+W]
            faces_found = detector.detect_faces(crop)
            if len(faces_found) == 1:
                res.append(crop)
    return res

def bad_or_good(image_path):
    img = imread(image_path)
    img = skimage.transform.resize(img, target_shape)
    img = np.dstack([img])
    feature = [embedding(tf.expand_dims(img, axis=0)).numpy()[0]]
    img_class = classifier.predict(feature)
    return img_class
    

# Define custom function for image management
def myImageHandler(bot, message, chat_id, image_name):
    print(image_name)
    temp_dir = tempfile.TemporaryDirectory()
    # send message to user
    bot.sendMessage(
        chat_id, f"Ciao utente {chat_id}!"
    )
    
    faces = find_faces(image_name)
    
    if len(faces) >= 1:
        for i, f in enumerate(faces):
            data = Image.fromarray(f)
            path = os.path.join(temp_dir.name, str(i)+".jpeg")
            print(path)
            data.save(path)
            bot.sendImage(chat_id, path, "Ecco una faccia!")
            result = bad_or_good(path)
            if result[0] == "savory":
                bot.sendMessage(chat_id, "La tua faccia è di uno/a buono/a")
            elif result[0] == "unsavory":
                bot.sendMessage(chat_id, "La tua faccia è di uno/a cattivo/a")
            else:
                bot.sendMessage(chat_id, "No funziona")
    else:
        bot.sendMessage(
            chat_id, "Faccia non trovata, spiaze...!"
        )


if __name__ == '__main__':
    # TODO: Load the model which will extract features from the given image
    bot = Updater(TOKEN)
    # TODO: Implement bot image handler
    bot.setPhotoHandler(myImageHandler)
    # Start telegram bot
    bot.start()