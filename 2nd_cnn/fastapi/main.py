## python fastapi tutorial https://lsjsj92.tistory.com/652 
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic import Field

app = FastAPI()

class DataInput(BaseModel):
    name: str

@app.get("/")
#async def root():
def root():
    return {"message" : "Hello World"}

@app.get("/home")
def home():
    return {"message" : "home"}

@app.get("/home/{name}")
def read_home(name:str):
    return {"message" : name}

@app.get("/home/{name}/err")
def read_home_err(name:int):
    return {"message" + str(NameError.name) : name}

@app.post("/")
def home_post(msg: DataInput):
    return {"Hello": "POST", "msg": msg.name}