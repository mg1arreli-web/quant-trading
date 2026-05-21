from backtester.circuit_breaker import CircuitBreaker


def test_circuit_breaker_halts():
    cb = CircuitBreaker(threshold=-0.08)

    # Mock SPY return of -0.08 (exactly threshold, triggers halt)
    halted = cb.check_returns(-0.08)
    assert halted is True
    assert cb.halted is True

    # Mock SPY return of -0.09 (below threshold, triggers halt)
    halted = cb.check_returns(-0.09)
    assert halted is True

def test_circuit_breaker_allows_trading():
    cb = CircuitBreaker(threshold=-0.08)

    # Mock SPY return of 0.01 (above threshold, allows trading)
    halted = cb.check_returns(0.01)
    assert halted is False
    assert cb.halted is False
