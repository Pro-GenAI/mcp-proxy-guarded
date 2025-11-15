"""Tests for the action classifier module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

from mcp_proxy_guarded.action_classifier import (
    ActionClassifier,
    get_classifier,
    is_action_harmful,
    SimpleMLP,
)


class TestSimpleMLP:
    """Test the SimpleMLP neural network."""

    def test_init(self):
        """Test SimpleMLP initialization."""
        model = SimpleMLP(in_dim=384, hidden=128)
        assert model.net is not None
        # Check that the first layer has the right input dimension
        assert model.net[0].in_features == 384
        assert model.net[0].out_features == 128

    def test_forward(self):
        """Test SimpleMLP forward pass."""
        model = SimpleMLP(in_dim=10, hidden=5)
        x = torch.randn(2, 10)
        output = model(x)
        assert output.shape == (2, 2)  # Binary classification


class TestActionClassifier:
    """Test the ActionClassifier class."""

    @patch('mcp_proxy.action_classifier.SentenceTransformer')
    @patch('mcp_proxy.action_classifier.torch.load')
    @patch('mcp_proxy.action_classifier.Path.exists')
    def test_init_with_model_loading(self, mock_exists, mock_torch_load, mock_sentence_transformer):
        """Test ActionClassifier initialization with model loading."""
        # Mock the model file exists
        mock_exists.return_value = True

        # Mock torch.load to return model state
        mock_checkpoint = {
            'in_dim': 384,
            'config': {'hidden': 128},
            'model_state_dict': {'net.0.weight': torch.randn(128, 384)},
        }
        mock_torch_load.return_value = mock_checkpoint

        # Mock SentenceTransformer
        mock_embedder = Mock()
        mock_sentence_transformer.return_value = mock_embedder

        classifier = ActionClassifier(load_model=True)

        assert classifier.embed_model == mock_embedder
        assert classifier.in_dim == 384
        assert classifier.model is not None
        mock_sentence_transformer.assert_called_once()
        mock_torch_load.assert_called_once()

    def test_init_without_model_loading(self):
        """Test ActionClassifier initialization without model loading."""
        classifier = ActionClassifier(load_model=False)

        assert classifier.embed_model is None
        assert classifier.model is None
        assert classifier.in_dim is None

    @patch('mcp_proxy.action_classifier.Path.exists')
    def test_load_model_file_not_found(self, mock_exists):
        """Test load_model when model file doesn't exist."""
        mock_exists.return_value = False

        classifier = ActionClassifier(load_model=False)
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            classifier.load_model()

    @patch('mcp_proxy.action_classifier.SentenceTransformer')
    def test_predict_without_model(self, mock_sentence_transformer):
        """Test predict method when model is not loaded."""
        classifier = ActionClassifier(load_model=False)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            classifier.predict({})

    @patch('mcp_proxy.action_classifier.SentenceTransformer')
    def test_predict_without_embedder(self, mock_sentence_transformer):
        """Test predict method when embedder is not initialized."""
        classifier = ActionClassifier(load_model=False)
        classifier.model = Mock()  # Mock the model

        with pytest.raises(RuntimeError, match="Embedding model not initialized"):
            classifier.predict({})

    @patch('mcp_proxy.action_classifier.SentenceTransformer')
    @patch('mcp_proxy.action_classifier.torch.load')
    @patch('mcp_proxy.action_classifier.Path.exists')
    def test_predict_success(self, mock_exists, mock_torch_load, mock_sentence_transformer):
        """Test successful prediction."""
        # Setup mocks
        mock_exists.return_value = True
        mock_checkpoint = {
            'in_dim': 10,
            'config': {'hidden': 5},
            'model_state_dict': {},
        }
        mock_torch_load.return_value = mock_checkpoint

        mock_embedder = Mock()
        mock_embedder.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
        mock_sentence_transformer.return_value = mock_embedder

        # Mock model
        mock_model = Mock()
        mock_logits = torch.tensor([[0.8, 0.2]])  # Higher probability for class 0
        mock_probs = torch.tensor([[0.8, 0.2]])
        mock_model.return_value = mock_logits

        classifier = ActionClassifier(load_model=True)
        classifier.model = mock_model

        # Mock torch functions
        with patch('torch.inference_mode'), \
             patch('torch.softmax', return_value=mock_probs) as mock_softmax, \
             patch('torch.argmax', return_value=torch.tensor([0])) as mock_argmax:

            action_dict = {
                "label": "test_action",
                "resource": "test_resource",
                "action": {
                    "server_label": "test_server",
                    "server_url": "http://test.com",
                    "parameters": {"param1": "value1"},
                    "require_approval": "never"
                }
            }

            is_harmful, confidence = classifier.predict(action_dict)

            assert is_harmful is False  # Class 0 is not harmful
            assert confidence == 0.8
            mock_embedder.encode.assert_called_once()

    def test_flatten_action_to_text(self):
        """Test the flatten_action_to_text method."""
        classifier = ActionClassifier(load_model=False)

        action_dict = {
            "label": "test_action",
            "resource": "test_resource",
            "action": {
                "server_label": "test_server",
                "server_url": "https://api.test.com/v1",
                "parameters": {"param1": "value1", "param2": "value2"},
                "require_approval": "never"
            }
        }

        result = classifier.flatten_action_to_text(action_dict)

        expected_parts = [
            "test_action",
            "test_resource",
            "test_server",
            "api.test.com",  # Host extracted from URL
            "never",
            "param1",  # Sorted parameter keys
            "param2"
        ]
        expected = " ".join(expected_parts)
        assert result == expected

    def test_flatten_action_to_text_no_url(self):
        """Test flatten_action_to_text with no server_url."""
        classifier = ActionClassifier(load_model=False)

        action_dict = {
            "label": "test_action",
            "resource": "test_resource",
            "action": {
                "server_label": "test_server",
                "server_url": "",
                "parameters": {},
                "require_approval": "yes"
            }
        }

        result = classifier.flatten_action_to_text(action_dict)
        expected = "test_action test_resource test_server  yes "
        assert result == expected


class TestGlobalClassifier:
    """Test global classifier functions."""

    @patch('mcp_proxy.action_classifier.ActionClassifier')
    def test_get_classifier(self, mock_classifier_class):
        """Test get_classifier function."""
        mock_instance = Mock()
        mock_classifier_class.return_value = mock_instance

        # First call should create instance
        result1 = get_classifier()
        assert result1 == mock_instance
        mock_classifier_class.assert_called_once()

        # Second call should return same instance
        result2 = get_classifier()
        assert result2 == mock_instance
        assert mock_classifier_class.call_count == 1  # Should not be called again

    @patch('mcp_proxy.action_classifier.get_classifier')
    def test_is_action_harmful(self, mock_get_classifier):
        """Test is_action_harmful convenience function."""
        mock_classifier = Mock()
        mock_classifier.predict.return_value = (True, 0.9)
        mock_get_classifier.return_value = mock_classifier

        result = is_action_harmful({"test": "data"})

        assert result == (True, 0.9)
        mock_classifier.predict.assert_called_once_with({"test": "data"})


class TestMainBlock:
    """Test the main block execution."""

    @patch('mcp_proxy.action_classifier.is_action_harmful')
    @patch('builtins.print')
    def test_main_execution(self, mock_print, mock_is_harmful):
        """Test the main block when run as script."""
        mock_is_harmful.return_value = (False, 0.3)

        # Import and run the main block
        import subprocess
        import sys

        # Run the module as a script
        result = subprocess.run([
            sys.executable, "-c",
            "from mcp_proxy.action_classifier import *; "
            "import sys; sys.path.insert(0, 'src'); "
            "exec(open('src/mcp_proxy/action_classifier.py').read().split('if __name__')[1])"
        ], capture_output=True, text=True, cwd="/home/praneeth/Desktop/mcp-proxy-guarded")

        # The main block should execute without errors
        assert result.returncode == 0
        mock_is_harmful.assert_called_once()