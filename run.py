from bot_server.Updater import Updater
from PIL import Image
import tempfile
import os
from src.DeepModel import DeepModel
import matplotlib.pyplot as plt


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

def imageSubplot(similar):
    count = 0
    x = len(similar)
    fig, axs = plt.subplots(2, 3)
    for i in range(0,2):
        for j in range(0,3):
            img = Image.open(similar[count])
            axs[i,j].axes.xaxis.set_visible(False)
            axs[i,j].axes.yaxis.set_visible(False)
            axs[i,j].title.set_text(str(count))
            axs[i,j].imshow(img)
            count +=1
    fig.savefig("temp.png")

    

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
            imageSubplot(most_similar)
            bot.sendImage(
                    chat_id, "./temp.png", "Faccie"
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