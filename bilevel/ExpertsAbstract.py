from abc import (
  ABC,
  abstractmethod,
)
#Abstract class for meta experts to be used by Adanormal hedge
class Expert(ABC):
    def __init__(self, name=None):
        self.name = name # can be OLS, Mult Weights etc

    @abstractmethod
    def get_ypred_t(self, t) -> float:
        '''
        Parameters
            t: Prediction for t th row of the data (X_t)
        
        Returns
            prediction in [0.0, 1.0] for the label
        '''
        raise NotImplementedError
    
    @abstractmethod
    def update_t(self, t) -> None:
        '''
            Updates the internal state of the expert
            return:
                None
        '''
        raise NotImplementedError
    
    @abstractmethod
    def cleanup(self) -> None:
        '''
            Deletes non essential member variables, at the end of all rounds 
        '''
        raise NotImplementedError 