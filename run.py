from bot_server.Updater import Updater
from PIL import Image
import tempfile
import os
from src.DeepModel import DeepModel
from src.BOVW import BOVW
from src.ColorHistogram import ColorHistogram
from src.CombinedModel import CombinedModel


# Siamese model paths
classifier_path = "./src/CNN_training/Classifier/svm_final_nu03.sav"
siamese_embeddings = "./src/CNN_training/final_siamese_embedding_model"
siamese_savory = "./src/CNN_training/Features/feature_savory.pkl"
siamese_unsavory = "./src/CNN_training/Features/feature_unsavory.pkl"
image_train_paths = "./src/CNN_training/Features/path_final.pkl"

# BOVW paths
bovw_path = './src/BOVW/bovw/bovw.pkl'
bovw_savory = './src/BOVW/bovw/feature_savory.pkl'
bovw_unsavory = './src/BOVW/bovw/feature_unsavory.pkl'
train_image_path = './src/BOVW/bovw/train_paths.pkl'

# Color paths
color_path = "./src/Color/color/histogram_model.pkl"
color_savory = "./src/Color/color/feature_savory.pkl"
color_unsavory = "./src/Color/color/feature_unsavory.pkl"

# Combined paths
combined_path = "./src/Combined_descriptors/combined/combined_model.pkl"

# BotFather Token to exploit created Telegram bot
TOKEN = "5758831231:AAFqcnYeS79nJ19ZwIoyJWbVv6Cbn6jSbnI"


def image_handler(tipo:str):
    def my_image_handler(bot, message, chat_id, name, image_name):
        print(image_name)
        temp_dir = tempfile.TemporaryDirectory()
        # send message to user
        bot.sendMessage(
            chat_id, f"{name}, tieniti forte, ora ti dirò se sei un birbantello!\nLasciami prima cercare il volto..."
        )
        
        faces = DeepModel.find_faces(image_name)
        
        if len(faces) >= 1:
            for i, f in enumerate(faces):
                image = Image.fromarray(f)
                path = os.path.join(temp_dir.name, str(i)+".jpeg")
                print(path)
                image.save(path)
                bot.sendImage(chat_id, path, "Ecco! Una faccia!")
                
                if tipo == "/Siamese":
                    result = model.predict(path)
                    most_similar_s, most_similar_u = model.cbir(image)
                elif tipo == "/BOVW":
                    result, most_similar_s, most_similar_u = BOVW.cbir(bovw_path, path, bovw_savory, bovw_unsavory, train_image_path)
                elif tipo == "/Color":
                    result, most_similar_s, most_similar_u = ColorHistogram.cbir(color_path, path, color_savory, color_unsavory, train_image_path)
                elif tipo == "/BOVWColor":
                    result, most_similar = CombinedModel.cbir(combined_path, path, None, None)
                    
                if result[0] == "savory":
                    bot.sendMessage(chat_id, "Complimenti! La tua faccia è buona! \N{smiling face with halo}")
                elif result[0] == "unsavory":
                    bot.sendMessage(chat_id, "Biricchino! La tua faccia è cattiva! \N{smiling face with horns}")
                else:
                    bot.sendMessage(chat_id, "Oops, qualcosa è andato storto! ")
                # CBIR
                bot.sendMessage(
                    chat_id, f"Ora, {name}, Ti farò vedere a chi assomigli di più!"
                )
                Updater.imageSubplot(most_similar_s, most_similar_u, temp_dir.name)
                bot.sendImage(
                    chat_id, os.path.join(temp_dir.name, "temp.png"), f"Ecco i {len(most_similar_s)*2} più simili!"
                )
        else:
            bot.sendMessage(
                chat_id, "Faccia non trovata, spiaze...! \N{face without mouth}"
            )
    return my_image_handler


def text_handler(theBot):
    def myTextHandler(bot, message, chat_id, name, text): 
        if text == "/BOVW" or text == "/Siamese" or text == "/Color" or text == "/BOVWColor":
            bot.sendMessage(
                chat_id, f"Eccomi {name}, quando sei pronto/a inviami una foto!"
            )
            theBot.setPhotoHandler(image_handler(tipo=text))
        else:
            bot.sendMessage(
                chat_id, f"Scusa {name}, ma non ho capito...\nEcco una lista di comandi:\n\t/BOVW\n\t/Siamese\n\t/Color\n\t/BOVWColor"
            )
    return myTextHandler


if __name__ == '__main__':
    # Load the model which will extract features from the given image
    model = DeepModel(
        classifier_path, 
        siamese_embeddings, 
        image_train_paths, 
        siamese_savory,
        siamese_unsavory, 
        (200, 200)
        )
    bot = Updater(TOKEN)
    
    # Implement command handler
    bot.setTextHandler(text_handler(bot))

    # Start telegram bot
    bot.start()