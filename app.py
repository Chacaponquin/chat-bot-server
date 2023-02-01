from fastapi import FastAPI

# routes
from modules.chat.routes.chatRoutes import chatRoutes

# app
app = FastAPI()

app.include_router(chatRoutes)
