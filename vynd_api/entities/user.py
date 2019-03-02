class User:
    _id: str 

    def __init__(self, id_: str):
        self.id = id_

    @property
    def id(self):
        return self._id
    