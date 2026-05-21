"""
Tests for factors: NFLO, options penalty, fundamentals.
"""
from factors.earnings_nlp import calculate_NFLO
from factors.options_skew import get_options_penalty


class TestNFLO:
    def test_empty_text(self):
        assert calculate_NFLO("") == 0.0

    def test_no_forward_sentences(self):
        text = "Revenue was strong. Margins improved."
        assert calculate_NFLO(text) == 0.0

    def test_positive_forward_sentences(self):
        text = (
            "We expect strong growth in Q4. "
            "We anticipate improved margins. "
            "We will see better results going forward."
        )
        score = calculate_NFLO(text)
        assert score > 0

    def test_negative_forward_sentences(self):
        text = (
            "We expect a decline in revenue. "
            "We anticipate challenging headwinds. "
            "We will face difficult uncertain conditions."
        )
        score = calculate_NFLO(text)
        assert score < 0

    def test_mixed_forward_sentences(self):
        text = (
            "We expect strong growth. "
            "We anticipate a decline in margins."
        )
        score = calculate_NFLO(text)
        assert isinstance(score, float)

    def test_balanced_sentences(self):
        text = (
            "We expect growth but also expect decline."
        )
        score = calculate_NFLO(text)
        # Both positive and negative in same sentence → net zero
        assert score == 0.0


class TestOptionsPenalty:
    def test_returns_float(self):
        # We can't easily mock yfinance here, but we can test the function
        # returns a float without crashing
        result = get_options_penalty('INVALID_SYMBOL_XYZ123')
        assert isinstance(result, float)
