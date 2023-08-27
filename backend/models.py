from pydantic import BaseModel
from typing import Tuple


class Address(BaseModel):
    target_building_id: int
    target_address: str
    # target_coordinates: Tuple[float, float]
