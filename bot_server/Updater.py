import logging
import time
import json, os, sys
import urllib
import tempfile
import requests
import matplotlib.pyplot as plt
from .Bot import Bot


def doNothing(*arg):
    pass

class Updater:
    def __init__(self, bot_id, waitingTime=0, download_folder=tempfile.gettempdir()+os.sep):
        self.bot = Bot(bot_id, download_folder)
        self.textHandler     = doNothing;
        self.photoHandler    = None
        self.RefineHandler   = doNothing;
        self.voiceHandler    = doNothing;
        self.documentHandler = doNothing;
        self.waitingTime     = waitingTime;

    def setTextHandler(self, f):
        self.textHandler = f

    def setPhotoHandler(self, f):
        self.photoHandler = f
        
    def setRefineHandler(self, f):
        self.RefineHandler = f

    def setVoiceHandler(self, f):
        self.voiceHandler = f

    def start(self):
        while True:
            for u in self.bot.getUpdates():
                # get info about the message
                messageType = self.bot.getMessageType(u['message'])
                message     = u['message']
                chat_id     = message['chat']['id']
                name        = message['chat']['first_name']
                message_id  = message['message_id']
                # call right functors
                if messageType == 'text':
                    text = message['text']
                    print("updater text:", text)
                    if text in ['/'+str(i) for i in range(1, 11)]:
                        print("settp refine")
                        if self.photoHandler is None:
                            self.bot.sendMessage(
                                chat_id, f"Scusa {name}, prima dovresti scegliere un metodo fra quelli a disposizione prima di poter raffinare la ricerca.\nEcco una lista di comandi:\n\t/BOVW\n\t/Color\n\t/BOVWColor\n\t/CNN"
                            )
                        else:
                            self.RefineHandler(self.bot, message, chat_id, name, text)
                    else:
                        print("settp text")
                        self.textHandler(self.bot, message, chat_id, name, text)
                if messageType == 'photo':
                    if self.photoHandler is None:
                        self.bot.sendMessage(
                            chat_id, f"Scusa {name}, prima dovresti scegliere un metodo fra quelli a disposizione.\nEcco una lista di comandi:\n\t/BOVW\n\t/Color\n\t/BOVWColor\n\t/CNN"
                        )
                    else:
                        local_filename = self.bot.getFile(u['message']['photo'][-1]['file_id'])
                        self.photoHandler(self.bot, message, chat_id, name, local_filename)
                if messageType == 'voice':
                    local_filename = self.bot.getFile(u['message']['voice']['file_id'])
                    self.voiceHandler(self.bot, message, chat_id, local_filename)
                if messageType == 'document':
                    local_filename = self.bot.getFile(u['message']['document']['file_id'])
                    self.documentHandler(self.bot, message, chat_id, local_filename)
            if self.waitingTime > 0:
                time.sleep(self.waitingTime)
                
                
    @staticmethod
    def imageSubplot(similar_s, similar_u, dir, dist_s, dist_u):
        import numpy as np
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 20))
        fig.suptitle('** I row: Savory - II row Unsavory **', fontsize=20)
        idx = np.argsort(np.argsort(np.concatenate((dist_s[0], dist_u[0]))))
        for i, path in enumerate(similar_s):
            plt.subplot(2, 5, i+1)
            plt.title(f"/{i+1} - Rank: {idx[i]+1} - dist: {np.round(dist_s[0][i], decimals=3)}")
            plt.axis('off')
            plt.imshow(plt.imread(path))
        for i, path in enumerate(similar_u):
            plt.subplot(2, 5, i+6)
            plt.title(f"/{i+6} - Rank: {idx[i+5]+1}- dist: {np.round(dist_u[0][i], decimals=3)}")
            plt.axis('off')
            plt.imshow(plt.imread(path))
        plt.savefig(os.path.join(dir, "temp.png"), bbox_inches='tight')
