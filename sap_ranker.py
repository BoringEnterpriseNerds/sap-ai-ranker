import random
import math
from typing import Dict, List, Tuple
from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()

class EloRanker:
    def __init__(self, items: List[str], k_factor: float = 32, min_comparisons: int = 10):
        """
        Initialize the Elo ranking system.
        
        Args:
            items: List of items to be ranked
            k_factor: How much each comparison affects the rating (default: 32)
            min_comparisons: Minimum number of comparisons per item (default: 10)
        """
        self.items = items
        self.k_factor = k_factor
        self.min_comparisons = min_comparisons
        self.ratings = {item: 1400 for item in items}  # Initial rating of 1400
        self.comparison_counts = {item: 0 for item in items}
        
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for item A when comparing with item B."""
        return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))
    
    def update_ratings(self, winner: str, loser: str) -> None:
        """Update ratings after a comparison."""
        expected_winner = self.expected_score(self.ratings[winner], self.ratings[loser])
        expected_loser = self.expected_score(self.ratings[loser], self.ratings[winner])
        
        # Update ratings
        self.ratings[winner] += self.k_factor * (1 - expected_winner)
        self.ratings[loser] += self.k_factor * (0 - expected_loser)
        
        # Update comparison counts
        self.comparison_counts[winner] += 1
        self.comparison_counts[loser] += 1
    
    def get_next_pair(self) -> Tuple[str, str]:
        """Get the next pair of items to compare."""
        # Prioritize items with fewer comparisons
        items_by_counts = sorted(self.items, key=lambda x: self.comparison_counts[x])
        item_a = items_by_counts[0]
        
        # Choose second item randomly from those with similar ratings
        remaining_items = [x for x in self.items if x != item_a]
        item_b = random.choice(remaining_items)
        
        return item_a, item_b
    
    def normalize_ratings(self) -> Dict[str, float]:
        """Normalize ratings to range from 0.1 to 0.9."""
        if not self.items:
            return {}
            
        # Get min and max ratings
        ratings_list = list(self.ratings.values())
        min_rating = min(ratings_list)
        max_rating = max(ratings_list)
        
        # Handle edge case where all ratings are the same
        if max_rating == min_rating:
            return {item: 0.5 for item in self.ratings}
        
        # Normalize ratings to 0.1-0.9 range
        normalized = {}
        for item, rating in self.ratings.items():
            normalized_value = 0.1 + 0.8 * (rating - min_rating) / (max_rating - min_rating)
            normalized[item] = normalized_value
            
        return normalized
    
    def is_ranking_confident(self) -> bool:
        """Check if we have enough comparisons for a confident ranking."""
        min_comparisons_met = all(count >= self.min_comparisons 
                                for count in self.comparison_counts.values())
        return min_comparisons_met

    def get_rankings(self, normalize: bool = False) -> List[Tuple[str, float, int]]:
        """
        Get current rankings sorted by rating.
        
        Args:
            normalize: If True, normalize ratings to 0.1-0.9 range
        """
        ratings_to_use = self.normalize_ratings() if normalize else self.ratings
        rankings = [(item, ratings_to_use[item], self.comparison_counts[item]) 
                   for item in self.items]
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    
    def perform_single_matchup(self, question_type, choice1, choice2):
        client = OpenAI()

        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "You are good at ranking things against one another. You're an expert in SAP development. "
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f"Think about the Gartner Magic Quadrant and rank these two SAP-related items on the {question_type} axis. Which is stronger: 1. {choice1}, or 2. {choice2}?"
                }
            ]
            },
            {
            "role": "assistant",
            "content": [
                {
                "type": "text",
                "text": "{\"value\":\"1\"}"
                }
            ]
            }
        ],
        temperature=1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
            "name": "single_string",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                "value": {
                    "type": "string",
                    "description": "A single string, either '1' or '2'.",
                    "enum": [
                    "1",
                    "2"
                    ]
                }
                },
                "required": [
                "value"
                ],
                "additionalProperties": False
            }
            }
        }
        )

        json_response = json.loads(response.choices[0].message.content)
        return json_response["value"]

def run_ranking_session():
    """Run an interactive ranking session."""
    # Get items from user
    print("Enter items to rank (one per line, empty line when done):")
    items = []
    while True:
        item = input().strip()
        if not item:
            break
        items.append(item)
    
    if len(items) < 2:
        print("Need at least 2 items to rank!")
        return
    

    
    for question in ["COMPLETENESS OF VISION", "ABILITY TO EXECUTE"]:
            # Initialize ranker
        ranker = EloRanker(items)
        comparison_count = 0
        # Main ranking loop
        while not ranker.is_ranking_confident():
            item_a, item_b = ranker.get_next_pair()
            comparison_count += 1
            
            while True:
                choice = ranker.perform_single_matchup(question, item_a, item_b)
                if choice in ('1', '2'):
                    break
            
            winner = item_a if choice == '1' else item_b
            loser = item_b if choice == '1' else item_a
            ranker.update_ratings(winner, loser)
            
        print(f"\nFinal rankings for {question}:")
        raw_rankings = ranker.get_rankings()
        normalized_rankings = ranker.get_rankings(normalize=True)
        
        for (rank, (item, raw_rating, count)), (_, (_, norm_rating, _)) in zip(enumerate(raw_rankings, 1), 
                                                                            enumerate(normalized_rankings, 1)):
            print(f"{rank}. {item}   Normalized (0.1-0.9): {norm_rating:.3f}")

if __name__ == "__main__":
    run_ranking_session()