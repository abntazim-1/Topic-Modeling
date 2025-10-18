"""
Example module demonstrating proper docstring formatting for MkDocs.

This module contains example classes and functions with comprehensive
docstrings that will be automatically converted to beautiful documentation
by mkdocstrings.
"""

from typing import List, Dict, Optional, Union, Tuple
import logging
from pathlib import Path


class ExampleProcessor:
    """
    A comprehensive example class demonstrating proper docstring formatting.
    
    This class showcases various docstring features that will be rendered
    beautifully in the MkDocs documentation, including:
    
    - Class descriptions
    - Parameter documentation
    - Return value documentation
    - Exception documentation
    - Usage examples
    
    Attributes:
        name (str): The name of the processor instance.
        config (Dict[str, any]): Configuration dictionary for the processor.
        logger (logging.Logger): Logger instance for this processor.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, any]] = None) -> None:
        """
        Initialize the ExampleProcessor.
        
        Args:
            name: A descriptive name for this processor instance.
            config: Optional configuration dictionary. If None, default
                configuration will be used.
                
        Example:
            >>> processor = ExampleProcessor("my_processor")
            >>> processor.name
            'my_processor'
        """
        self.name = name
        self.config = config or {"default": True}
        self.logger = logging.getLogger(f"processor.{name}")
        
    def process_text(self, text: str, options: Optional[Dict[str, any]] = None) -> Dict[str, any]:
        """
        Process input text according to the specified options.
        
        This method demonstrates comprehensive parameter and return value
        documentation that will be automatically formatted in the docs.
        
        Args:
            text: The input text to process. Must be a non-empty string.
            options: Optional processing options. Supported keys:
                - 'lowercase': Convert text to lowercase (default: True)
                - 'remove_punctuation': Remove punctuation marks (default: False)
                - 'max_length': Maximum output length (default: None)
                
        Returns:
            A dictionary containing:
                - 'processed_text': The processed text string
                - 'original_length': Length of original text
                - 'processed_length': Length of processed text
                - 'options_used': The options that were applied
                
        Raises:
            ValueError: If text is empty or None.
            TypeError: If text is not a string.
            
        Example:
            >>> processor = ExampleProcessor("test")
            >>> result = processor.process_text("Hello, World!")
            >>> result['processed_text']
            'hello, world!'
        """
        if not text:
            raise ValueError("Text cannot be empty or None")
        if not isinstance(text, str):
            raise TypeError("Text must be a string")
            
        options = options or {}
        lowercase = options.get('lowercase', True)
        remove_punctuation = options.get('remove_punctuation', False)
        max_length = options.get('max_length')
        
        processed = text
        if lowercase:
            processed = processed.lower()
        if remove_punctuation:
            import string
            processed = processed.translate(str.maketrans('', '', string.punctuation))
        if max_length and len(processed) > max_length:
            processed = processed[:max_length]
            
        return {
            'processed_text': processed,
            'original_length': len(text),
            'processed_length': len(processed),
            'options_used': {
                'lowercase': lowercase,
                'remove_punctuation': remove_punctuation,
                'max_length': max_length
            }
        }
    
    def batch_process(self, texts: List[str], **kwargs) -> List[Dict[str, any]]:
        """
        Process multiple texts in batch.
        
        Args:
            texts: List of text strings to process.
            **kwargs: Additional options passed to process_text.
            
        Returns:
            List of processing results, one for each input text.
            
        Example:
            >>> processor = ExampleProcessor("batch")
            >>> texts = ["Hello", "World", "Python"]
            >>> results = processor.batch_process(texts, lowercase=True)
            >>> len(results)
            3
        """
        return [self.process_text(text, kwargs) for text in texts]
    
    def save_results(self, results: List[Dict[str, any]], output_path: Union[str, Path]) -> None:
        """
        Save processing results to a file.
        
        Args:
            results: List of processing result dictionaries.
            output_path: Path where to save the results.
            
        Raises:
            IOError: If the file cannot be written.
        """
        import json
        output_path = Path(output_path)
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise IOError(f"Cannot write to {output_path}: {e}")


def utility_function(text: str, multiplier: int = 2) -> str:
    """
    A utility function demonstrating function-level documentation.
    
    This function shows how to document standalone functions that will
    appear in the API reference.
    
    Args:
        text: Input text to process.
        multiplier: Number of times to repeat the text.
        
    Returns:
        The input text repeated 'multiplier' times.
        
    Example:
        >>> utility_function("hello", 3)
        'hellohellohello'
    """
    return text * multiplier


def advanced_function(
    data: List[Dict[str, any]], 
    filter_key: str, 
    filter_value: any,
    sort_by: Optional[str] = None
) -> Tuple[List[Dict[str, any]], int]:
    """
    Advanced function with complex parameters and return types.
    
    This function demonstrates documentation for more complex function
    signatures with multiple parameters and tuple return types.
    
    Args:
        data: List of dictionaries to filter and optionally sort.
        filter_key: Key to filter dictionaries by.
        filter_value: Value to match against the filter_key.
        sort_by: Optional key to sort results by. If None, no sorting.
        
    Returns:
        A tuple containing:
            - Filtered (and optionally sorted) list of dictionaries
            - Count of items that matched the filter
            
    Example:
        >>> data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        >>> filtered, count = advanced_function(data, "age", 30)
        >>> count
        1
        >>> filtered[0]["name"]
        'Alice'
    """
    filtered = [item for item in data if item.get(filter_key) == filter_value]
    
    if sort_by:
        filtered.sort(key=lambda x: x.get(sort_by, 0))
    
    return filtered, len(filtered)


# Module-level constants and variables
DEFAULT_CONFIG = {
    "lowercase": True,
    "remove_punctuation": False,
    "max_length": None
}

SUPPORTED_LANGUAGES = ["en", "es", "fr", "de"]

# This will be documented as a module-level variable
version = "1.0.0"