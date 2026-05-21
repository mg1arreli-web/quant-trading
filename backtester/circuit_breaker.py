from config.settings import CIRCUIT_BREAKER_THRESHOLD


class CircuitBreaker:
    def __init__(self, threshold=CIRCUIT_BREAKER_THRESHOLD):
        self.threshold = threshold
        self.halted = False

    def check_returns(self, current_return):
        """
        Check if the current return triggers the circuit breaker.
        Updates and returns the halted flag.
        """
        if current_return <= self.threshold:
            self.halted = True
        else:
            self.halted = False
        return self.halted
