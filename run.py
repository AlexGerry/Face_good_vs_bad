from bot_server.Updater import Updater
from PIL import Image
import tempfile
import os
from src.DeepModel import DeepModel


classifier_path = "./src/CNN_training/Classifier/svm_final_nu03.sav"
siamese_embeddings = "./src/CNN_training/final_siamese_embedding_model"
image_train_features = "./src/CNN_training/Features/feature_final.pkl"
image_train_paths = "./src/CNN_training/Features/path_final.pkl"
# Load the model which will extract features from the given image
model = DeepModel(
    classifier_path, 
    siamese_embeddings, 
    image_train_paths, 
    image_train_features, 
    (200,200)
    )

# BotFather Token to exploit created Telegram bot
TOKEN = "5758831231:AAFqcnYeS79nJ19ZwIoyJWbVv6Cbn6jSbnI"
    

# Define custom function for image management
def myImageHandler(bot, message, chat_id, name, image_name):
    print(image_name)
    temp_dir = tempfile.TemporaryDirectory()
    # send message to user
    bot.sendMessage(
        chat_id, f"Ciao {name}, ora ti dirò se sei un birbantello!"
    )
    
    faces = DeepModel.find_faces(image_name)
    
    if len(faces) >= 1:
        for i, f in enumerate(faces):
            image = Image.fromarray(f)
            path = os.path.join(temp_dir.name, str(i)+".jpeg")
            print(path)
            image.save(path)
            bot.sendImage(chat_id, path, "Ecco una faccia!")
            # Predict good or bad image
            result = model.predict(path)
            if result[0] == "savory":
                bot.sendMessage(chat_id, "La tua faccia è di uno/a buono/a")
            elif result[0] == "unsavory":
                bot.sendMessage(chat_id, "La tua faccia è di uno/a cattivo/a")
            else:
                bot.sendMessage(chat_id, "No funziona")
            # TODO: CBIR
            bot.sendMessage(
                chat_id, f"Ora, {name}, Ti farò vedere a chi assomigli di più!"
            )
            most_similar = model.cbir(image)
            bot.sendMessage(
                chat_id, f"Most similar: {str(most_similar)}"
            )
    else:
        bot.sendMessage(
            chat_id, "Faccia non trovata, spiaze...!"
        )


if __name__ == '__main__':
    bot = Updater(TOKEN)
    # Implement bot image handler
    bot.setPhotoHandler(myImageHandler)
    # Start telegram bot
    bot.start()