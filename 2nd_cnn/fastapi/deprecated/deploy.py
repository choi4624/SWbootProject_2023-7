from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import subprocess


app = FastAPI()

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


# Assuming you have a PyTorch model saved in 'model.ckpt'
model = torch.load('/home/i4624/vscode/gitclone/SWbootProject_2023-7/2nd_cnn/fastapi/results/patchcore/mvtec/hazelnut/run/weights/model.ckpt')
model.eval()


# Define the input data model for the uploaded image
class InputDataModel(model):
    file: UploadFile



def preprocess_image(file):
    # Implement image preprocessing here (if needed)
    return processed_image

def predict(input_data):
    try:
        # Preprocess the uploaded image
        input_image = preprocess_image(input_data.file)
        
        # Assuming 'model' is the loaded model
        # Make predictions using the model
        # Replace 'result' with the actual prediction result



        result = model(input_image)

        return result
    except Exception as e:
        # Return an error response if something goes wrong
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/predict/")
async def predict_endpoint(input_data: InputDataModel):
    try:
        # Save the uploaded image to a temporary file
        image_path = f"/uploaded/image.jpg"
        with open(image_path, "wb") as f:
            f.write(await input_data.file.read())

        # Define the command to run the external script
        cmd = [
            "python3",
            "/home/i4624/vscode/gitclone/SWbootProject_2023-7/2nd_cnn/anomalib2/tools/inference/lightning_inference.py",
            "--config",
            "/home/i4624/vscode/gitclone/SWbootProject_2023-7/2nd_cnn/anomalib/src/anomalib/models/patchcore/config.yaml",
            "--weights",
            "/home/i4624/vscode/gitclone/SWbootProject_2023-7/2nd_cnn/fastapi/results/patchcore/mvtec/hazelnut/run/weights/model.ckpt",
            "--input",
            image_path,
            "--output",
            "./infer_results",
            "--visualization_mode",
            "full",
        ]

        # Execute the external script using subprocess
        subprocess.run(cmd, check=True, capture_output=True)

        # Read the output file of the external script (modify this according to your output format)
        with open("/path/to/output/file.txt", "r") as output_file:
            prediction_result = output_file.read()

        return {"prediction": prediction_result}
    except Exception as e:
        # Return an error response if something goes wrong
        return JSONResponse(status_code=500, content={"error": str(e)})
