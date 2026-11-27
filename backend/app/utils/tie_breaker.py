"""
Tie-Breaking Logic Module for 4-Model Hard Majority Voting Ensemble.

Academic Specification:
In a 4-classifier ensemble, each model casts one binary vote (0 = Ham, 1 = Spam).
A 2-2 tie occurs when exactly two models vote Spam and two models vote Ham.

Tie-Breaking Rule (as required):
1. If there is a 2-2 tie, calculate the average calibrated spam probability of all four models.
2. If the average spam probability >= 0.5, predict Spam (1).
3. If the average spam probability < 0.5, predict Ham (0).
"""

from typing import Dict, Any, Optional


def resolve_tie(
    votes: Dict[str, int],
    probabilities: Dict[str, float],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluates and resolves a potential 2-2 voting tie using the average calibrated spam probability.

    Args:
        votes: Dictionary mapping model identifier to binary prediction (0: Ham, 1: Spam).
        probabilities: Dictionary mapping model identifier to spam probability P(spam).
        threshold: Decision threshold for average spam probability (default 0.5).

    Returns:
        Dictionary containing:
        - is_tie: bool (True if exactly 2-2 tie)
        - tie_resolved: bool
        - average_spam_probability: float
        - resolved_prediction: int (0 or 1)
        - resolved_label: str ('ham' or 'spam')
        - explanation: str (Human-readable rationale)
    """
    spam_votes = sum(1 for v in votes.values() if v == 1)
    ham_votes = sum(1 for v in votes.values() if v == 0)

    # Check if this is indeed a 2-2 tie
    if spam_votes != 2 or ham_votes != 2:
        return {
            "is_tie": False,
            "tie_resolved": False,
            "average_spam_probability": None,
            "resolved_prediction": None,
            "resolved_label": None,
            "explanation": f"No tie detected (Spam votes: {spam_votes}, Ham votes: {ham_votes})."
        }

    # Calculate average calibrated spam probability across all models
    model_probs = list(probabilities.values())
    if not model_probs:
        avg_prob = 0.0
    else:
        avg_prob = sum(model_probs) / len(model_probs)

    resolved_pred = 1 if avg_prob >= threshold else 0
    resolved_label = "spam" if resolved_pred == 1 else "ham"

    explanation = (
        f"2-2 voting tie detected (2 Spam vs 2 Ham). "
        f"Invoked tie-breaker: calculated average calibrated spam probability across all 4 models "
        f"= {avg_prob:.4f} (Threshold = {threshold:.2f}). "
        f"Since {avg_prob:.4f} {'is >= 0.5' if avg_prob >= threshold else 'is < 0.5'}, "
        f"tie was resolved as {resolved_label.upper()}."
    )

    return {
        "is_tie": True,
        "tie_resolved": True,
        "average_spam_probability": round(float(avg_prob), 4),
        "resolved_prediction": resolved_pred,
        "resolved_label": resolved_label,
        "threshold": threshold,
        "explanation": explanation
    }
