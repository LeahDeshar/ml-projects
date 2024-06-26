import io
import pickle 
import numpy as np  
import PIL.Image
import PIL.ImageOps
from PIL import Image
from fastapi import FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware


with open('mnist_model.pkl','rb') as f:
    model = pickle.load(f)
    
    
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/predict-image/')
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    pil_image= PIL.Image.open(io.BytesIO(contents)).convert('L')
    pil_image = PIL.ImageOps.invert(pil_image)
    pil_image = pil_image.resize((28,28),PIL.Image.ANTIALIAS)
    img_array = np.array(pil_image).reshape(1,-1)
    prediction = model.predict(img_array)
    return {"prediction": int(prediction[0])}





import pickle
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


X,y = fetch_openml('mnist_784',version=1,return_X_y=True)

# test and train
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# classifier
clf = RandomForestClassifier(n_jobs=-1)

# print
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

with open('mnist_model.pkl','wb') as f:
    pickle.dump(clf,f)