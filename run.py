from bot_server.Updater import Updater
from PIL import Image
import tempfile
import os
from .src.DeepModel import DeepModel


model = DeepModel("./src/CNN_training/Classifier/svm2.sav", "./src/CNN_training/siamese_embedding_model", (200,200))

# BotFather Token to exploit created Telegram bot
TOKEN = "5758831231:AAFqcnYeS79nJ19ZwIoyJWbVv6Cbn6jSbnI"
    

# Define custom function for image management
def myImageHandler(bot, message, chat_id, image_name):
    print(image_name)
    temp_dir = tempfile.TemporaryDirectory()
    # send message to user
    bot.sendMessage(
        chat_id, f"Ciao utente {chat_id}!"
    )
    
    faces = DeepModel.find_faces(image_name)
    
    if len(faces) >= 1:
        for i, f in enumerate(faces):
            data = Image.fromarray(f)
            path = os.path.join(temp_dir.name, str(i)+".jpeg")
            print(path)
            data.save(path)
            bot.sendImage(chat_id, path, "Ecco una faccia!")
            
            result = model.predict(path)
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