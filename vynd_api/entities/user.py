class User:
    __id: str 

    def __init__(self, id_: str):
        self.__id = id_

    @property
    def id(self):
        return self.__id
    