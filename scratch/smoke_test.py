import sys
from unittest.mock import patch
from app.cli import main

inputs = [
    "2401.12345", 
    "How does the proposed method work?", 
    "What is the weather in tokyo?", 
    "exit"
]

def mock_input(prompt):
    print(prompt, end="")
    val = inputs.pop(0)
    print(val)
    return val

with patch("builtins.input", mock_input):
    main()
