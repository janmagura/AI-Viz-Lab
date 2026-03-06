"""
AI-Viz-Lab: Configuration Manager
Handles loading and validating configuration from YAML files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    """Manages application configuration."""
    
    DEFAULT_CONFIG = {
        'hardware': {
            'device': 'cpu',
            'threads': 4,
            'max_memory_gb': 8
        },
        'model': {
            'path': 'Qwen/Qwen2.5-1.5B-Instruct',
            'precision': 'float32',
            'max_new_tokens': 64
        },
        'performance': {
            'enable_profiling': False,
            'log_to_file': False
        },
        'ui': {
            'theme': 'dark',
            'animations': True
        }
    }
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            print(f"⚠️  Config file not found: {self.config_path}")
            print("   Using default configuration")
            return self.DEFAULT_CONFIG.copy()
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Merge with defaults
            merged = self.DEFAULT_CONFIG.copy()
            self._deep_merge(merged, config)
            
            print(f"✅ Loaded configuration from {self.config_path}")
            return merged
            
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            print("   Using default configuration")
            return self.DEFAULT_CONFIG.copy()
    
    def _deep_merge(self, base: dict, override: dict):
        """Deep merge two dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def get(self, *keys, default=None):
        """Get nested configuration value."""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def validate(self) -> bool:
        """Validate configuration values."""
        valid = True
        
        # Check device
        device = self.get('hardware', 'device')
        if device not in ['cpu', 'cuda', 'mps']:
            print(f"⚠️  Invalid device: {device}. Must be cpu, cuda, or mps")
            valid = False
        
        # Check precision
        precision = self.get('model', 'precision')
        if precision not in ['float32', 'float16', 'int8', 'int4']:
            print(f"⚠️  Invalid precision: {precision}")
            valid = False
        
        # Check threads
        threads = self.get('hardware', 'threads')
        if threads < 1:
            print(f"⚠️  Invalid threads: {threads}. Must be >= 1")
            valid = False
        
        if valid:
            print("✅ Configuration validated successfully")
        
        return valid
    
    def save(self, path: Optional[str] = None):
        """Save current configuration to file."""
        output_path = Path(path) if path else self.config_path
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        print(f"✅ Configuration saved to {output_path}")


# Global config instance
_config_manager: Optional[ConfigManager] = None

def get_config() -> ConfigManager:
    """Get global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def reload_config(config_path: str = "config.yaml") -> ConfigManager:
    """Reload configuration from file."""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager
