from bot_server.Updater import Updater


# BotFather Token to exploit created Telegram bot
TOKEN = "5758831231:AAFqcnYeS79nJ19ZwIoyJWbVv6Cbn6jSbnI"


# Define custom function for image management
def myImageHandler(bot, message, chat_id, text):
    # TODO: ...
    pass

if __name__ == '__main__':
    # TODO: Load the model which will extract features from the given image
    bot = Updater(TOKEN)
    # TODO: Implement bot image handler
    bot.setPhotoHandler(myImageHandler)
    # Start telegram bot
    bot.start()