from bot_server.Updater import Updater
from PIL import Image
import tempfile
import os
import numpy as np
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
color_path = "./src/Color/color_withus/histogram_model.pkl"
color_savory = "./src/Color/color_withus/feature_savory.pkl"
color_unsavory = "./src/Color/color_withus/feature_unsavory.pkl"

# Combined paths
combined_path = "./src/Combined_descriptors/combined/combined_model.pkl"

# BotFather Token to exploit created Telegram bot
TOKEN = "5758831231:AAFqcnYeS79nJ19ZwIoyJWbVv6Cbn6jSbnI"

# Track similar retrieved images by user id to refine search
# Keys -> User Chat ID
# Values -> [model type, img query feature, path imgs results]
similar_by_chatId = {}


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
                    most_similar_s, most_similar_u, feature, dist_s, dist_u = model.cbir(image)
                elif tipo == "/BOVW":
                    result, most_similar_s, most_similar_u, feature, dist_s, dist_u = BOVW.cbir(bovw_path, path, bovw_savory, bovw_unsavory, train_image_path)
                elif tipo == "/Color":
                    result, most_similar_s, most_similar_u, feature, dist_s, dist_u = ColorHistogram.cbir(color_path, path, color_savory, color_unsavory, train_image_path)
                elif tipo == "/BOVWColor":
                    result, most_similar_s, most_similar_u, feature, dist_s, dist_u = CombinedModel.cbir(combined_path, path, None, None)
                    
                similar_by_chatId[chat_id] = [tipo, feature, np.concatenate((most_similar_s, most_similar_u))] # -> tipo modello utilizzato, img query feature, img result
                print("similar_by_chatId:", similar_by_chatId)
                
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
                Updater.imageSubplot(most_similar_s, most_similar_u, temp_dir.name, dist_s, dist_u)
                bot.sendImage(
                    chat_id, os.path.join(temp_dir.name, "temp.png"), f"Ecco i {len(most_similar_s)*2} più simili!"
                )
        else:
            bot.sendMessage(
                chat_id, "Faccia non trovata, spiaze...! \N{face without mouth}"
            )
    return my_image_handler


def refineSearch_handler(theBot):
    def my_refine_handler(bot, message, chat_id, name, text):
        print("refine_handler text:", text)
        temp_dir = tempfile.TemporaryDirectory()
        if chat_id not in similar_by_chatId:
            print("error")
            bot.sendMessage(
                chat_id, f"Perdonami {name}, ma c'è stato un errore..."
            )
        else:
            bot.sendMessage(
                chat_id, f"{name}, adesso raffinerò la ricerca in base al tuo suggerimento..."
            )
            image_idx = int(text[-1])
            model_type, query_img_feature, query_res = similar_by_chatId[chat_id]
            print(model_type, query_img_feature, query_res)
            selected_img = query_res[image_idx-1]
            print(selected_img)
            # Get mean embedding between query img and selected img
            if model_type == "/Siamese":
                mean_emb, most_similar_s, most_similar_u, dist_s, dist_u = model.refine_search(query_img_feature, selected_img)
            elif model_type == "/BOVW":
                mean_emb, most_similar_s, most_similar_u, dist_s, dist_u = BOVW.refine_search(query_img_feature, selected_img, bovw_path, bovw_savory, bovw_unsavory, train_image_path)
            elif model_type == "/Color":
                mean_emb, most_similar_s, most_similar_u, dist_s, dist_u = ColorHistogram.refine_search(query_img_feature, selected_img, color_path, color_savory, color_unsavory, train_image_path)
            elif model_type == "/BOVWColor":
                mean_emb, most_similar_s, most_similar_u, dist_s, dist_u = CombinedModel.refine_search(query_img_feature, selected_img, combined_path, None, None)
            
            # Update similar for iterative refinements
            similar_by_chatId[chat_id][1] = mean_emb
            similar_by_chatId[chat_id][2] = np.concatenate((most_similar_s, most_similar_u))
            
            Updater.imageSubplot(most_similar_s, most_similar_u, temp_dir.name, dist_s, dist_u)
            bot.sendImage(
                chat_id, os.path.join(temp_dir.name, "temp.png"), f"Ecco i {len(most_similar_s)*2} più simili dopo il raffinamento!"
            )
        
    return my_refine_handler


def text_handler(theBot):
    def myTextHandler(bot, message, chat_id, name, text): 
        print(text)
        print("similar_by_chatId:", similar_by_chatId)
        if text == "/BOVW" or text == "/Siamese" or text == "/Color" or text == "/BOVWColor":
            bot.sendMessage(
                chat_id, f"Eccomi {name}, quando sei pronto/a inviami una foto!"
            )
            theBot.setPhotoHandler(image_handler(tipo=text))
        else:
            bot.sendMessage(
                chat_id, 
                f"Scusa {name}, ma non ho capito...\
                \nEcco una lista di comandi:\n\t/BOVW\n\t/Siamese\n\t/Color\n\t/BOVWColor\n\n \
                Oppure, dopo aver utilizzato un metodo, scrivi / seguito da un numero (1-10) per raffinare la ricerca."
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
    # Implement refine handler
    bot.setRefineHandler(refineSearch_handler(bot))

    # Start telegram bot
    bot.start()