import uvicorn
import json

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from models import Address
from typing import List, Annotated

from v3 import AddressMatcher

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])

weights = {
    "post_prefix_building": 0.9,
    "full_address_building": 0.8,
    "name_prefix": 0.6,
    "house_building": 0.7,
    "corpus_building": 0.4,
    "build_number_building": 0.5,
    "liter_building": 0.1
}

train_data_path = 'result.csv'
address_matcher = AddressMatcher(weights)
address_matcher.load_train_data(train_data_path)


@app.get("/search", description="Возвращает наиболее подходящие адреса под запрос")
def search(q: str) -> List[Address]:
    # обращение к модели
    result = json.loads(address_matcher.find_matching_addresses(q, num_results=10))['result']
    # for i in range(1, 6):
    #     result.append(
    #         Address(target_building_id=i, target_address=f"{q} {i}",
    #                 target_coordinates=[12.64645646 * i, 31.6456654 * i]))

    return result


@app.post("/upload_dataset", description="Загружает датасет")
def upload_dataset(file: Annotated[bytes, File()] = None):
    if not file:
        return {"message": "No file sent"}
    else:
        return {"file_size": len(file)}


if __name__ == '__main__':
    uvicorn.run(app, port=5000)
