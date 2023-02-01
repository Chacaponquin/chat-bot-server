from fastapi import APIRouter

# dto
from modules.chat.dto.messageDTO import MessageDTO

# services
from modules.neuronalNetwork.services.nnServices import getResponseFromMessage, trainModel

chatRoutes = APIRouter(prefix='/chat')


@chatRoutes.post('/newMessage')
def chatWithBot(message: MessageDTO):
    msg = dict(message)
    resultMessage = getResponseFromMessage(msg['message'])
    return resultMessage
