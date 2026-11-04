import numpy as np

from app.main import app
from app.utils.preprocessing import preprocess_for_efficientnet, preprocess_for_mesonet
from fastapi.testclient import TestClient


client = TestClient(app)


def test_app_root_route():
    response = client.get("/")
    assert response.status_code == 200


def test_efficientnet_preprocess_returns_tensor_and_face_status():
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    tensor, face_detected = preprocess_for_efficientnet(image)

    assert tensor.shape == (1, 3, 224, 224)
    assert isinstance(face_detected, bool)


def test_mesonet_preprocess_returns_tensor_and_face_status():
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    tensor, face_detected = preprocess_for_mesonet(image)

    assert tensor.shape == (1, 3, 256, 256)
    assert isinstance(face_detected, bool)
