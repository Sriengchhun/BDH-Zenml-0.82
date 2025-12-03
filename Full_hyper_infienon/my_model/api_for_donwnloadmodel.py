import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

app = FastAPI()

@app.get("/download")
async def download_file(file_name: str):
    # Append the .pkl extension to the provided file name
    full_file_name = f"{file_name}.pkl"
    
    # Construct the file path
    # file_path = os.path.join(os.getcwd(), full_file_name)
    # Construct the file path in the specific directory
    file_path = os.path.join("/app/Full_hyper_infienon/my_model", full_file_name)
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Return the file as a FileResponse
    return FileResponse(path=file_path, media_type='application/octet-stream', filename=full_file_name)


# from fastapi import FastAPI
# from fastapi.responses import FileResponse

# file_path = "model_2024-05-03.pkl"
# app = FastAPI()

# @app.get("/")
# def main():
#     return FileResponse(path=file_path, filename=file_path, media_type='text/mp4')

# import os

# from fastapi import FastAPI
# from fastapi.responses import FileResponse


# app = FastAPI()


# @app.get("/")
# async def main():
#     file_name = "model_2024-05-03.pkl"
#     # DEPENDS ON WHERE YOUR FILE LOCATES
#     file_path = os.getcwd() + "/" + file_name
#     return FileResponse(path=file_path, media_type='application/octet-stream', filename=file_name)





##uvicorn api_for_donwnloadmodel:app --host 0.0.0.0 --port 8999 --reload