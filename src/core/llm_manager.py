"""
Multi-LLM Model Manager
Handles integration with multiple AI models
"""

import os
import json
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class LLMModel:
    name: str
    provider: str
    api_key: Optional[str]
    endpoint: str
    status: str
    model_id: str
    cost_per_1k_tokens: float
    capabilities: List[str]

class LLMManager:
    def __init__(self, config_path="llm_config.json"):
        self.config_path = config_path
        self.models: Dict[str, LLMModel] = {}
        self.load_config()

    def load_config(self):
        """Load LLM configurations"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                for name, data in config.get('models', {}).items():
                    self.models[name] = LLMModel(**data)
        else:
            self.create_default_config()

    def create_default_config(self):
        """Create default configuration"""
        default_models = {
            "Claude 3.5 Sonnet": LLMModel(
                name="Claude 3.5 Sonnet",
                provider="Anthropic",
                api_key=os.getenv("ANTHROPIC_API_KEY", ""),
                endpoint="https://api.anthropic.com/v1/messages",
                status="Ready",
                model_id="claude-3-5-sonnet-20241022",
                cost_per_1k_tokens=0.003,
                capabilities=["code", "analysis", "creative", "vision"]
            ),
            "Claude 3 Opus": LLMModel(
                name="Claude 3 Opus",
                provider="Anthropic",
                api_key=os.getenv("ANTHROPIC_API_KEY", ""),
                endpoint="https://api.anthropic.com/v1/messages",
                status="Ready",
                model_id="claude-3-opus-20240229",
                cost_per_1k_tokens=0.015,
                capabilities=["code", "analysis", "creative", "vision", "advanced"]
            ),
            "GPT-4": LLMModel(
                name="GPT-4",
                provider="OpenAI",
                api_key=os.getenv("OPENAI_API_KEY", ""),
                endpoint="https://api.openai.com/v1/chat/completions",
                status="Ready",
                model_id="gpt-4",
                cost_per_1k_tokens=0.03,
                capabilities=["code", "analysis", "creative"]
            ),
            "GPT-3.5 Turbo": LLMModel(
                name="GPT-3.5 Turbo",
                provider="OpenAI",
                api_key=os.getenv("OPENAI_API_KEY", ""),
                endpoint="https://api.openai.com/v1/chat/completions",
                status="Ready",
                model_id="gpt-3.5-turbo",
                cost_per_1k_tokens=0.0005,
                capabilities=["code", "analysis", "fast"]
            ),
            "Gemini Pro": LLMModel(
                name="Gemini Pro",
                provider="Google",
                api_key=os.getenv("GOOGLE_API_KEY", ""),
                endpoint="https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent",
                status="Ready",
                model_id="gemini-pro",
                cost_per_1k_tokens=0.00025,
                capabilities=["code", "analysis", "multimodal"]
            ),
            "LLaMA 3": LLMModel(
                name="LLaMA 3",
                provider="Meta",
                api_key="",
                endpoint="http://localhost:11434/api/generate",  # Ollama local
                status="Ready",
                model_id="llama3",
                cost_per_1k_tokens=0.0,  # Local/free
                capabilities=["code", "analysis", "local"]
            ),
            "Mistral AI": LLMModel(
                name="Mistral AI",
                provider="Mistral AI",
                api_key=os.getenv("MISTRAL_API_KEY", ""),
                endpoint="https://api.mistral.ai/v1/chat/completions",
                status="Ready",
                model_id="mistral-large-latest",
                cost_per_1k_tokens=0.002,
                capabilities=["code", "analysis", "multilingual"]
            )
        }

        self.models = default_models
        self.save_config()

    def save_config(self):
        """Save configuration to file"""
        config = {
            "models": {
                name: {
                    "name": model.name,
                    "provider": model.provider,
                    "api_key": model.api_key,
                    "endpoint": model.endpoint,
                    "status": model.status,
                    "model_id": model.model_id,
                    "cost_per_1k_tokens": model.cost_per_1k_tokens,
                    "capabilities": model.capabilities
                }
                for name, model in self.models.items()
            }
        }

        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def query_claude(self, prompt: str, model_name: str = "Claude 3.5 Sonnet") -> dict:
        """Query Claude models"""
        model = self.models.get(model_name)
        if not model or not model.api_key:
            return {"error": "Model not configured or API key missing"}

        try:
            headers = {
                "x-api-key": model.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            data = {
                "model": model.model_id,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}]
            }

            response = requests.post(model.endpoint, headers=headers, json=data)
            return response.json()

        except Exception as e:
            return {"error": str(e)}

    def query_openai(self, prompt: str, model_name: str = "GPT-3.5 Turbo") -> dict:
        """Query OpenAI models"""
        model = self.models.get(model_name)
        if not model or not model.api_key:
            return {"error": "Model not configured or API key missing"}

        try:
            headers = {
                "Authorization": f"Bearer {model.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": model.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024
            }

            response = requests.post(model.endpoint, headers=headers, json=data)
            return response.json()

        except Exception as e:
            return {"error": str(e)}

    def query_gemini(self, prompt: str) -> dict:
        """Query Google Gemini"""
        model = self.models.get("Gemini Pro")
        if not model or not model.api_key:
            return {"error": "Model not configured or API key missing"}

        try:
            url = f"{model.endpoint}?key={model.api_key}"

            data = {
                "contents": [{"parts": [{"text": prompt}]}]
            }

            response = requests.post(url, json=data)
            return response.json()

        except Exception as e:
            return {"error": str(e)}

    def query_ollama(self, prompt: str, model_name: str = "LLaMA 3") -> dict:
        """Query local Ollama models"""
        model = self.models.get(model_name)
        if not model:
            return {"error": "Model not configured"}

        try:
            data = {
                "model": model.model_id,
                "prompt": prompt,
                "stream": False
            }

            response = requests.post(model.endpoint, json=data)
            return response.json()

        except Exception as e:
            return {"error": f"Ollama not running. Install with: curl https://ollama.ai/install.sh | sh\nThen run: ollama pull {model.model_id}"}

    def query_mistral(self, prompt: str) -> dict:
        """Query Mistral AI"""
        model = self.models.get("Mistral AI")
        if not model or not model.api_key:
            return {"error": "Model not configured or API key missing"}

        try:
            headers = {
                "Authorization": f"Bearer {model.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": model.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024
            }

            response = requests.post(model.endpoint, headers=headers, json=data)
            return response.json()

        except Exception as e:
            return {"error": str(e)}

    def query_model(self, prompt: str, model_name: str) -> dict:
        """Query any model by name"""
        model = self.models.get(model_name)
        if not model:
            return {"error": "Model not found"}

        # Route to appropriate query function
        if "Claude" in model_name:
            return self.query_claude(prompt, model_name)
        elif "GPT" in model_name:
            return self.query_openai(prompt, model_name)
        elif "Gemini" in model_name:
            return self.query_gemini(prompt)
        elif "LLaMA" in model_name:
            return self.query_ollama(prompt, model_name)
        elif "Mistral" in model_name:
            return self.query_mistral(prompt)
        else:
            return {"error": "Unknown model provider"}

    def download_local_models(self):
        """Download local models via Ollama"""
        models_to_download = [
            ("llama3", "Meta LLaMA 3 - 8B parameters"),
            ("mistral", "Mistral 7B - Efficient and fast"),
            ("codellama", "Code LLaMA - Specialized for coding"),
            ("gemma", "Google Gemma - 7B parameters"),
            ("phi", "Microsoft Phi-2 - Compact model")
        ]

        return models_to_download

    def get_model_status(self) -> Dict[str, dict]:
        """Get status of all models"""
        status = {}
        for name, model in self.models.items():
            has_key = bool(model.api_key) if model.api_key else (model.provider == "Meta")  # Local doesn't need key
            status[name] = {
                "provider": model.provider,
                "configured": has_key,
                "status": model.status if has_key else "Not Configured",
                "cost_per_1k": model.cost_per_1k_tokens,
                "capabilities": model.capabilities
            }

        return status

    def add_custom_model(self, name: str, provider: str, endpoint: str,
                        model_id: str, api_key: str = "") -> bool:
        """Add a custom model"""
        try:
            self.models[name] = LLMModel(
                name=name,
                provider=provider,
                api_key=api_key,
                endpoint=endpoint,
                status="Ready",
                model_id=model_id,
                cost_per_1k_tokens=0.0,
                capabilities=["custom"]
            )
            self.save_config()
            return True
        except Exception as e:
            print(f"Error adding model: {e}")
            return False

    def update_api_key(self, model_name: str, api_key: str) -> bool:
        """Update API key for a model"""
        if model_name in self.models:
            self.models[model_name].api_key = api_key
            self.models[model_name].status = "Ready"
            self.save_config()
            return True
        return False

# Example usage
if __name__ == "__main__":
    manager = LLMManager()

    print("Available Models:")
    print("=" * 60)

    status = manager.get_model_status()
    for name, info in status.items():
        configured = "✓" if info["configured"] else "✗"
        print(f"{configured} {name}")
        print(f"  Provider: {info['provider']}")
        print(f"  Status: {info['status']}")
        print(f"  Cost/1K tokens: ${info['cost_per_1k']}")
        print(f"  Capabilities: {', '.join(info['capabilities'])}")
        print()

    print("\nLocal Models Available for Download:")
    print("=" * 60)
    for model_id, description in manager.download_local_models():
        print(f"- {model_id}: {description}")
        print(f"  Download: ollama pull {model_id}")
        print()
