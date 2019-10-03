from dataclasses import dataclass
from typing import List


@dataclass
class Product:
    """
    Data class for products.
    """
    name: str
    price: int
    # description: str
    image: str
    # labels: List[str]
    # price: float
    # created_at: int
    id: str = None


    @staticmethod
    def deserialize(document):
        """
        Helper function for parsing a Firestore document to a Product object.

        Parameters:
           document (DocumentSnapshot): A snapshot of Firestore document.

        Output:
           A Product object.
        """
        data = document.to_dict()
        if data:
            return Product(
                id=document.id,
                name=data.get('name'),
                # description=data.get('description'),
                image=data.get('image'),
                price=data.get('price')
                # labels=data.get('labels'),
                # price=data.get('price'),
                # created_at=data.get('created_at')
            )

        return None