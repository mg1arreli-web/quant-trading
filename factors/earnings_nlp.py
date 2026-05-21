"""
Earnings NLP Factor — NFLO (Net Forward-Looking Optimism) score from transcripts.
"""
import logging

from data_pipeline.sec_transcript_fetcher import fetch_earnings_transcript

logger = logging.getLogger(__name__)


def calculate_NFLO(text: str) -> float:
    """
    Net Forward-Looking Optimism (NFLO) calculator.
    NFLO = (Positive Forward Sentences - Negative Forward Sentences) / Total Forward Sentences
    """
    if not text:
        return 0.0

    forward_words = ['will', 'expect', 'anticipate', 'future', 'guidance', 'forecast', 'project', 'plan', 'forward']
    positive_words = [
        'growth', 'strong', 'increase', 'improve', 'record',
        'better', 'confident', 'optimistic', 'success', 'benefit',
    ]
    negative_words = [
        'decline', 'decrease', 'weak', 'challenge', 'difficult',
        'risk', 'uncertain', 'headwind', 'lower', 'worse',
    ]

    sentences = text.split('.')
    forward_sentences = [s.strip().lower() for s in sentences if any(fw in s.lower() for fw in forward_words)]

    if not forward_sentences:
        return 0.0

    pos_count = 0
    neg_count = 0
    for s in forward_sentences:
        if any(pw in s for pw in positive_words):
            pos_count += 1
        if any(nw in s for nw in negative_words):
            neg_count += 1

    total_forward = len(forward_sentences)
    if total_forward == 0:
        return 0.0

    nflo = (pos_count - neg_count) / total_forward
    return nflo


def get_earnings_nlp_score(symbol: str, year: int = 2024) -> float:
    """Fetch transcript and compute NFLO score."""
    transcript = fetch_earnings_transcript(symbol, year)
    if not transcript:
        logger.debug(f"No transcript available for {symbol}")
        return 0.0
    score = calculate_NFLO(transcript)
    logger.debug(f"NFLO score for {symbol}: {score:.4f}")
    return score
