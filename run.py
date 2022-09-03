from bot_server.Updater import Updater
from PIL import Image
import tempfile
import os
import numpy as np
#from src.DeepModel import DeepModel
from src.BOVW import BOVW
from src.ColorHistogram import ColorHistogram
from src.CombinedModel import CombinedModel
from src.CNN import CNN
from src.Preprocess import find_face_and_preprocess
from mtcnn import MTCNN


# Siamese model paths
#classifier_path = "./src/CNN_training/Classifier_withus/svm_final_nu03.sav"
#siamese_embeddings = "./src/CNN_training/final_siamese_embedding_model_withus"
#siamese_savory = "./src/CNN_training/Features_withus/feature_savory.pkl"
#siamese_unsavory = "./src/CNN_training/Features_withus/feature_unsavory.pkl"
#image_train_paths = "./src/CNN_training/Features_withus/path_final.pkl"

# BOVW paths
bovw_path = './src/BOVW/bovw_withus/bovw.pkl'
bovw_savory = './src/BOVW/bovw_withus/feature_savory.pkl'
bovw_unsavory = './src/BOVW/bovw_withus/feature_unsavory.pkl'
train_image_path = './src/BOVW/bovw_withus/train_paths.pkl'

# Color paths
color_path = "./src/Color/color_withus/histogram_model.pkl"
color_savory = "./src/Color/color_withus/feature_savory.pkl"
color_unsavory = "./src/Color/color_withus/feature_unsavory.pkl"

# Combined paths
combined_withus_path = "./src/Combined_descriptors/combined_withus/combined_model.pkl"

# CNN
cnn_path = './src/CNN/trained_cnn/'
cnn_savory = './src/CNN/cnn_withus_maxpool/feature_savory.pkl'
cnn_unsavory = './src/CNN/cnn_withus_maxpool/feature_unsavory.pkl'
cnn_image_train_paths = "./src/CNN/cnn_withus_maxpool/cnn_train_paths.pkl"

# BotFather Token to exploit created Telegram bot
TOKEN = "5758831231:AAFqcnYeS79nJ19ZwIoyJWbVv6Cbn6jSbnI"

# Track similar retrieved images by user id to refine search
# Keys -> User Chat ID
# Values -> [model type, img query feature, path imgs results, refine counter]
similar_by_chatId = {}


def image_handler(tipo:str):
    def my_image_handler(bot, message, chat_id, name, image_name):
        print(image_name)
        temp_dir = tempfile.TemporaryDirectory()
        # send message to user
        bot.sendMessage(
            chat_id, f"{name}, tieniti forte, ora ti dirò se sei un birbantello!\nLasciami prima cercare il volto..."
        )
        
        faces = find_face_and_preprocess(image_name, detector)
        
        if len(faces) >= 1:
            for i, f in enumerate(faces):
                f = (f*255).astype('uint8')
                image = Image.fromarray(f)
                path = os.path.join(temp_dir.name, str(i)+".jpeg")
                print(path)
                image.save(path)
                bot.sendImage(chat_id, path, "Ecco! Una faccia!")
                score = None
                #if tipo == "/Siamese":
                #    result = model.predict(path)
                #    most_similar_s, most_similar_u, feature, dist_s, dist_u = model.cbir(image)
                if tipo == "/BOVW":
                    score, result, most_similar_s, most_similar_u, feature, dist_s, dist_u = BOVW.cbir(bovw_path, path, bovw_savory, bovw_unsavory, train_image_path)
                elif tipo == "/Color":
                    score, result, most_similar_s, most_similar_u, feature, dist_s, dist_u = ColorHistogram.cbir(color_path, path, color_savory, color_unsavory, train_image_path)
                elif tipo == "/BOVWColor":
                    score, result, most_similar_s, most_similar_u, feature, dist_s, dist_u = CombinedModel.cbir(combined_withus_path, path, None, train_image_path)
                elif tipo == "/CNN":
                    score, res, most_similar_s, most_similar_u, feature, dist_s, dist_u = cnn.cbir(path, cnn_savory, cnn_unsavory, cnn_image_train_paths)
                    result = []
                    if res[0] == 0:
                        result.append('savory')
                    elif res[0] == 1:
                        result.append('unsavory')
                    
                similar_by_chatId[chat_id] = [tipo, feature, np.concatenate((most_similar_s, most_similar_u)), 1] # -> tipo modello utilizzato, img query feature, img result, refine counter
                print("similar_by_chatId:", similar_by_chatId)
                
                if result[0] == "savory":
                    bot.sendMessage(chat_id, "Complimenti! La tua faccia è buona! \N{smiling face with halo}")
                elif result[0] == "unsavory":
                    bot.sendMessage(chat_id, "Biricchino! La tua faccia è cattiva! \N{smiling face with horns}")
                else:
                    bot.sendMessage(chat_id, "Oops, qualcosa è andato storto! ")
                    
                if score is not None:
                    bot.sendMessage(chat_id, f"Ne sono certo al {np.round(np.max(score), decimals=2)*100}%!")
                    
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
            model_type, query_img_feature, query_res, refine_counter = similar_by_chatId[chat_id]
            print(model_type, query_img_feature, query_res, refine_counter)
            selected_img = query_res[image_idx-1]
            # Get mean embedding between query img and selected img
            #if model_type == "/Siamese":
            #    mean_emb, most_similar_s, most_similar_u, dist_s, dist_u = model.refine_search(refine_counter, query_img_feature, selected_img)
            if model_type == "/BOVW":
                mean_emb, most_similar_s, most_similar_u, dist_s, dist_u = BOVW.refine_search(refine_counter, query_img_feature, selected_img, bovw_path, bovw_savory, bovw_unsavory, train_image_path)
            elif model_type == "/Color":
                mean_emb, most_similar_s, most_similar_u, dist_s, dist_u = ColorHistogram.refine_search(refine_counter, query_img_feature, selected_img, color_path, color_savory, color_unsavory, train_image_path)
            elif model_type == "/BOVWColor":
                mean_emb, most_similar_s, most_similar_u, dist_s, dist_u = CombinedModel.refine_search(refine_counter, query_img_feature, selected_img, combined_withus_path, None, train_image_path)
            elif model_type == "/CNN":
                mean_emb, most_similar_s, most_similar_u, dist_s, dist_u = cnn.refine_search(refine_counter, query_img_feature, selected_img, cnn_savory, cnn_unsavory, cnn_image_train_paths)

            # Update similar for iterative refinements
            similar_by_chatId[chat_id][1] = mean_emb
            similar_by_chatId[chat_id][2] = np.concatenate((most_similar_s, most_similar_u))
            similar_by_chatId[chat_id][3] += 1
            
            Updater.imageSubplot(most_similar_s, most_similar_u, temp_dir.name, dist_s, dist_u)
            bot.sendImage(
                chat_id, os.path.join(temp_dir.name, "temp.png"), f"Ecco i {len(most_similar_s)*2} più simili dopo il raffinamento!"
            )
        
    return my_refine_handler


def text_handler(theBot):
    def myTextHandler(bot, message, chat_id, name, text): 
        print("similar_by_chatId:", similar_by_chatId)
        if text == "/BOVW" or text == "/Color" or text == "/BOVWColor" or text == "/CNN":
            bot.sendMessage(
                chat_id, f"Eccomi {name}, quando sei pronto/a inviami una foto!"
            )
            theBot.setPhotoHandler(image_handler(tipo=text))
        else:
            bot.sendMessage(
                chat_id, 
                f"Scusa {name}, ma non ho capito...\
                \nEcco una lista di comandi:\n\t/BOVW\n\t/Color\n\t/BOVWColor\n\t/CNN\n\n \
                Oppure, dopo aver utilizzato un metodo, scrivi / seguito da un numero (1-10) per raffinare la ricerca."
            )
    return myTextHandler


if __name__ == '__main__':
    # Load the model which will extract features from the given image
    #model = DeepModel(
    #    classifier_path, 
    #    siamese_embeddings, 
    #    image_train_paths, 
    #    siamese_savory,
    #    siamese_unsavory, 
    #    (200, 200)
    #    )
    detector = MTCNN()
    cnn = CNN(cnn_path)
    
    bot = Updater(TOKEN)
    
    # Implement command handler
    bot.setTextHandler(text_handler(bot))
    # Implement refine handler
    bot.setRefineHandler(refineSearch_handler(bot))

    # Start telegram bot
    bot.start()