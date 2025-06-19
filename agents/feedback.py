# === Scoring Agent ===
class ScoringAgent:
    @staticmethod
    def generate_feedback(memory: dict) -> dict:
        # Dummy scoring
        return {
            "overall_score": 4.0,
            "feedback": "You communicated clearly and showed decent problem-solving skills."
        }