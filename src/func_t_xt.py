from abc import abstractmethod

import typing

class Function_t_xt() :

    @abstractmethod
    def apply(t: float, xt: float) -> float:
        pass

