from bot_server.Updater import Updater
from PIL import Image
import tempfile
import os
from src.DeepModel import DeepModel
from src.BOVW import BOVW
import matplotlib.pyplot as plt


# Siamese model paths
classifier_path = "./src/CNN_training/Classifier/svm_final_nu03.sav"
siamese_embeddings = "./src/CNN_training/final_siamese_embedding_model"
image_train_features = "./src/CNN_training/Features/feature_final.pkl"
image_train_paths = "./src/CNN_training/Features/path_final.pkl"

# BOVW paths
bovw_path = './src/BOVW/bovw/bovw.pkl'
train_voc_path = './src/BOVW/bovw/train_bovw.pkl'
train_image_path = './src/BOVW/bovw/train_paths.pkl'

# BotFather Token to exploit created Telegram bot
TOKEN = "5758831231:AAFqcnYeS79nJ19ZwIoyJWbVv6Cbn6jSbnI"


def imageSubplot(similar, dir):
    fig, axes = plt.subplots(nrows=len(similar), ncols=1, figsize=(20, 20))
    for i, path in enumerate(similar):
        plt.subplot(len(similar), 1, i+1)
        plt.title(f"{i+1}")
        plt.axis('off')
        plt.imshow(plt.imread(path))
    plt.savefig(os.path.join(dir, "temp.png"), bbox_inches='tight')


# Define custom function for image management
#def myImageHandler_Siamese(bot, message, chat_id, name, image_name):
#    print(image_name)
#    temp_dir = tempfile.TemporaryDirectory()
#    # send message to user
#    bot.sendMessage(
#        chat_id, f"Ciao {name}, ora ti dirò se sei un birbantello!\nLasciami prima cercare il volto..."
#    )
#    
#    faces = DeepModel.find_faces(image_name)
#    
#    if len(faces) >= 1:
#        for i, f in enumerate(faces):
#            image = Image.fromarray(f)
#            path = os.path.join(temp_dir.name, str(i)+".jpeg")
#            print(path)
#            image.save(path)
#            bot.sendImage(chat_id, path, "Ecco! una faccia!")
#            # Predict good or bad image
#            result = model.predict(path)
#            if result[0] == "savory":
#                bot.sendMessage(chat_id, "La tua faccia è di uno/a buono/a")
#            elif result[0] == "unsavory":
#                bot.sendMessage(chat_id, "La tua faccia è di uno/a cattivo/a")
#            else:
#                bot.sendMessage(chat_id, "No funziona")
#            # CBIR
#            bot.sendMessage(
#                chat_id, f"Ora, {name}, Ti farò vedere a chi assomigli di più!"
#            )
#            most_similar = model.cbir(image, 6)
#            imageSubplot(most_similar, temp_dir.name)
#            bot.sendImage(
#                chat_id, os.path.join(temp_dir.name, "temp.png"), f"Ecco i {len(most_similar)} più simili!"
#            )
#    else:
#        bot.sendMessage(
#            chat_id, "Faccia non trovata, spiaze...!"
#        )
        

def myImageHandler_BOVW(bot, message, chat_id, name, image_name):
    print(image_name)
    temp_dir = tempfile.TemporaryDirectory()
    # send message to user
    bot.sendMessage(
        chat_id, f"Ciao {name}, ora ti dirò se sei un birbantello!\nLasciami prima cercare il volto..."
    )
    
    faces = DeepModel.find_faces(image_name)
    
    if len(faces) >= 1:
        for i, f in enumerate(faces):
            image = Image.fromarray(f)
            path = os.path.join(temp_dir.name, str(i)+".jpeg")
            print(path)
            image.save(path)
            bot.sendImage(chat_id, path, "Ecco! Una faccia!")
            # Predict good or bad image
            pred, most_similar = BOVW.cbir(bovw_path, path, train_voc_path, train_image_path)
            if pred == "savory":
                bot.sendMessage(chat_id, "La tua faccia è di uno/a buono/a")
            elif pred == "unsavory":
                bot.sendMessage(chat_id, "La tua faccia è di uno/a cattivo/a")
            else:
                bot.sendMessage(chat_id, "No funziona")
            # CBIR
            imageSubplot(most_similar, temp_dir.name)
            bot.sendImage(
                chat_id, os.path.join(temp_dir.name, "temp.png"), f"Ecco i {len(most_similar)} più simili!"
            )
    else:
        bot.sendMessage(
            chat_id, "Faccia non trovata, spiaze...!"
        )


if __name__ == '__main__':
    # Load the model which will extract features from the given image
    #model = DeepModel(
    #    classifier_path, 
    #    siamese_embeddings, 
    #    image_train_paths, 
    #    image_train_features, 
    #    (200, 200)
    #    )
    bot = Updater(TOKEN)
    # Implement bot image handler
    bot.setPhotoHandler(myImageHandler_BOVW)
    # Start telegram bot
    bot.start()