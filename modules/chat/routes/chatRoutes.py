from fastapi import APIRouter

# dto
from modules.chat.dto.messageDTO import MessageDTO

# services
from modules.neuronalNetwork.services.nnServices import getResponseFromMessage

chatRoutes = APIRouter(prefix='/chat')


@chatRoutes.post('/newMessage')
def chatWithBot(message: MessageDTO):
    msg = dict(message)
    return getResponseFromMessage(msg['message'])
