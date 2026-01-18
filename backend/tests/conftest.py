import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import sys
import os

# Ensure src is in pythonpath
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the app (but don't start it yet)
from src.inference_edge import app, ml_resources

@pytest.fixture(scope="module", autouse=True)
def mock_heavy_lifting():
    """
    Globally patch the heavy ML libraries.
    This runs BEFORE any test or app startup.
    """
    print("\n>>> 🛡️ PATCHING ML MODELS TO PREVENT OOM...")
    
    with patch("src.inference_edge.ORTModelForCausalLM") as mock_ort, \
         patch("src.inference_edge.AutoTokenizer") as mock_tok:
        
        # 1. Setup Mock Model
        mock_model_instance = MagicMock()
        # Mock the .generate() method to return a dummy tensor
        # We simulate a token sequence: [input_len + generated_len]
        # Shape: (1, 10)
        mock_model_instance.generate.return_value = [[101, 202, 303, 404, 505]]
        
        mock_ort.from_pretrained.return_value = mock_model_instance

        # 2. Setup Mock Tokenizer
        mock_tok_instance = MagicMock()
        mock_tok_instance.apply_chat_template.return_value = "Mocked Prompt"
        
        # --- CRITICAL FIX: Make the tokenizer output an OBJECT, not a Dict ---
        mock_tokenizer_output = MagicMock()
        # Give it an .input_ids attribute with a .shape property
        mock_tokenizer_output.input_ids.shape = (1, 2) 
        
        # When tokenizer(...) is called, return this object
        mock_tok_instance.return_value = mock_tokenizer_output
        
        # Mock .decode
        mock_tok_instance.decode.return_value = "CRITICAL FAULT: Mocked System Failure"
        
        mock_tok.from_pretrained.return_value = mock_tok_instance

        yield mock_ort, mock_tok

@pytest.fixture(scope="module")
def client():
    """
    TestClient that uses the mocked app context.
    """
    # The 'with' block triggers the lifespan startup
    with TestClient(app) as c:
        yield c
    
    # Cleanup global resources after tests
    ml_resources.clear()