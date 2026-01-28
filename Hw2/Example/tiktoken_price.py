"""
API Cost Calculation Example

What this does:
    Shows how to calculate how much an OpenAI API call will cost based on tokens used.
    This function helps you estimate costs before making a call, or calculate the
    exact cost after a call. Different models have different prices, so this helps
    you manage your budget.

What you'll need:
    Nothing! This is a standalone function that only uses basic Python.

What you'll learn:
    - How to look up pricing for different GPT models
    - Why input (prompt) and output (completion) tokens cost different amounts
    - How to calculate the total cost from token counts
    - How to handle models that aren't in the pricing list
    - How to track costs in real-time as you use the API
"""

def calculate_api_cost(prompt_tokens: int, completion_tokens: int, model_name: str) -> float:
    pricing = {
        "gpt-4o": {
            "prompt": 0.005 / 1000, # $0.005/1M 
            "completion": 0.015 / 1000 # $0.015/1M
        },
        "gpt-3.5-turbo": {
            "prompt": 0.0005 / 1000, # $0.50/1M
            "completion": 0.0015 / 1000 # $1.50/1M
        }
    }

    if model_name in pricing:
        prompt_cost = prompt_tokens * pricing[model_name]["prompt"]
        completion_cost = completion_tokens * pricing[model_name]["completion"]
        total_cost = prompt_cost + completion_cost
        return total_cost
    else:
        raise ValueError(f"Pricing for model {model_name} not found.")

model_name = "gpt-4o"
my_prompt_tokens = 15
my_completion_tokens = 500 # Note you have some control, over this to set upper-bound...

total_cost = calculate_api_cost(my_prompt_tokens, my_completion_tokens, model_name)
print(f"The estimated cost for this API call is ${total_cost:.6f}")
