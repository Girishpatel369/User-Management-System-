

import telepot


def sendtoTelegram( msg ):

    token = '7986819134:AAH8rYURAdM3yj6m3kLE9NDXjHyamODc16g' # telegram token
    receiver_id = 901107151 # https://api.telegram.org/bot<TOKEN>/getUpdates


    bot = telepot.Bot(token)

    bot.sendMessage(receiver_id, msg) 
    bot.sendPhoto(receiver_id, photo=open('./demo.jpg', 'rb'))



def sendtoTelegram1( msg ):

    token = '7986819134:AAH8rYURAdM3yj6m3kLE9NDXjHyamODc16g' # telegram token
    receiver_id = 901107151 # https://api.telegram.org/bot<TOKEN>/getUpdates


    bot = telepot.Bot(token)

    bot.sendMessage(receiver_id, msg) 
    bot.sendPhoto(receiver_id, photo=open('./oneway-violation.jpg', 'rb'))



# sendtoTelegram("Helmetviolation detected")