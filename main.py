from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import numpy as np
import requests
import threading
from contextlib import asynccontextmanager
from tensorflow.keras.models import load_model

model = None
_lock = threading.Lock()


def get_model():
    global model
    if model is None:
        with _lock:  # stops double-loading if two requests hit at once
            if model is None:
                model = load_model("plant_disease_classification.keras")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load in background so the port binds instantly
    threading.Thread(target=get_model, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

class_names = ["Healthy", "Powdery", "Rust"]

MAX_SIZE = 10 * 1024 * 1024  # 10 MB
HEADERS = {"User-Agent": "Mozilla/5.0 (PlantAI Bot)"}


def render(request: Request, **context):
    return templates.TemplateResponse(
        request=request, name="index.html", context=context
    )


@app.get("/")
def home(request: Request):
    return render(request)


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(None),
    image_url: str = Form(""),
):
    image_bytes = None
    source_name = ""

    # ---------- GET IMAGE ----------
    try:
        if file is not None and file.filename:
            image_bytes = await file.read()
            source_name = file.filename

        elif image_url.strip():
            url = image_url.strip()
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()

            ctype = resp.headers.get("Content-Type", "")
            if not ctype.startswith("image/"):
                return render(
                    request,
                    error="That URL is not a direct image link. Use a link ending in .jpg / .png.",
                )
            image_bytes = resp.content
            source_name = url

        else:
            return render(request, error="Please upload an image or enter an image URL.")

    except requests.exceptions.Timeout:
        return render(request, error="The image URL took too long to respond.")
    except requests.exceptions.RequestException:
        return render(request, error="Could not download the image from that URL.")
    except Exception:
        return render(request, error="Could not read the image. Try again.")

    if not image_bytes:
        return render(request, error="Empty image received.")
    if len(image_bytes) > MAX_SIZE:
        return render(request, error="Image is too large (max 10 MB).")

    # ---------- PROCESS + PREDICT ----------
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img = img.resize((225, 225))

        img_array = np.array(img, dtype="float32") / 255.0
        img_array = img_array.reshape(1, 225, 225, 3)

        pred = get_model().predict(img_array, verbose=0)

        predicted_class = class_names[int(np.argmax(pred))]
        confidence = float(np.max(pred)) * 100

    except UnidentifiedImageError:
        return render(request, error="That file is not a valid image.")
    except Exception:
        return render(request, error="Something went wrong while analyzing the image.")

    return render(
        request,
        prediction=predicted_class,
        confidence=round(confidence, 2),
        filename=source_name,
    )