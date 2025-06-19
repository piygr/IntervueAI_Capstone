import logging

logger = logging.getLogger(__name__)


# === Heuristic Engine ===
class HeuristicEngine:
    @staticmethod
    def passed(user_text: str) -> bool:
        return MinWordCountHeuristic.check(user_text) and ResponseTimeHeuristic.check()

class MinWordCountHeuristic:
    @staticmethod
    def check(text):
        return len(text.split()) >= 5

class ResponseTimeHeuristic:
    @staticmethod
    def check():
        # Placeholder: could be implemented via session timestamp tracking
        return True