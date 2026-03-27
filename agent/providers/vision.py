"""Vision provider for image analysis and generation.

Extends the Google provider with vision (image understanding) and
Nano Banana (image generation) capabilities.
"""
import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union

from google import genai
from google.genai import types
from PIL import Image

logger = logging.getLogger(__name__)


class VisionProvider:
    """Handles image analysis and generation via Gemini."""

    def __init__(self, api_key: str):
        """Initialize vision provider.
        
        Args:
            api_key: Google API key
        """
        self.client = genai.Client(api_key=api_key)
        self.vision_model = "gemini-3-flash-preview"  # For image understanding
        self.generation_model = "gemini-3.1-flash-image-preview"  # Nano Banana
        
    async def analyze_image(
        self,
        image_data: Union[bytes, str, Path],
        prompt: str = "Describe this image in detail.",
        mime_type: str = "image/jpeg"
    ) -> str:
        """Analyze an image using Gemini's vision capabilities.
        
        Args:
            image_data: Image as bytes, base64 string, or file path
            prompt: Analysis prompt
            mime_type: Image MIME type (image/jpeg, image/png, etc.)
            
        Returns:
            Analysis result as text
        """
        try:
            # Handle different input types
            if isinstance(image_data, (str, Path)):
                # File path
                with open(image_data, 'rb') as f:
                    image_bytes = f.read()
            elif isinstance(image_data, str) and image_data.startswith('data:'):
                # Data URI - extract base64
                _, base64_data = image_data.split(',', 1)
                image_bytes = base64.b64decode(base64_data)
            else:
                # Already bytes
                image_bytes = image_data

            # Create image part
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            )

            # Generate analysis
            response = self.client.models.generate_content(
                model=self.vision_model,
                contents=[image_part, prompt]
            )
            
            return response.text

        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            raise

    async def analyze_multiple_images(
        self,
        images: List[Dict[str, Union[bytes, str]]],
        prompt: str
    ) -> str:
        """Analyze multiple images together.
        
        Args:
            images: List of dicts with 'data' (bytes) and 'mime_type' keys
            prompt: Analysis prompt
            
        Returns:
            Combined analysis result
        """
        try:
            parts = []
            
            # Add all images
            for img in images:
                parts.append(
                    types.Part.from_bytes(
                        data=img['data'],
                        mime_type=img.get('mime_type', 'image/jpeg')
                    )
                )
            
            # Add prompt
            parts.append(prompt)
            
            response = self.client.models.generate_content(
                model=self.vision_model,
                contents=parts
            )
            
            return response.text

        except Exception as e:
            logger.error(f"Multi-image analysis failed: {e}")
            raise

    async def generate_image(
        self,
        prompt: str,
        num_images: int = 1,
        aspect_ratio: str = "1:1",
        safety_filter: str = "default"
    ) -> List[bytes]:
        """Generate images using Nano Banana (Gemini's image generation).
        
        Args:
            prompt: Image generation prompt
            num_images: Number of images to generate (1-4)
            aspect_ratio: "1:1", "16:9", "9:16", "4:3", "3:4"
            safety_filter: "default", "strict", "permissive"
            
        Returns:
            List of image bytes
        """
        try:
            config = types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            )
            
            # Enhanced prompt with aspect ratio hint
            enhanced_prompt = prompt
            if aspect_ratio != "1:1":
                enhanced_prompt = f"{prompt} ({aspect_ratio} aspect ratio)"
            
            response = self.client.models.generate_content(
                model=self.generation_model,
                contents=enhanced_prompt,
                config=config
            )
            
            # Extract generated images - handle different response formats
            images = []
            
            # Check if response has candidates
            if not response.candidates:
                logger.error("No candidates in response")
                raise ValueError("No image generated - empty response")
            
            candidate = response.candidates[0]
            
            # The response format from Nano Banana can vary
            # Try different extraction methods
            
            # Method 1: Check content.parts for inline_data
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        # inline_data.data might be base64 string or already bytes
                        data = part.inline_data.data
                        if isinstance(data, str):
                            # Base64 encoded
                            image_bytes = base64.b64decode(data)
                        else:
                            # Already bytes
                            image_bytes = data
                        images.append(image_bytes)
                        logger.info(f"Extracted image via inline_data: {len(image_bytes)} bytes")
            
            # Method 2: Check for blob/file data (alternative format)
            if not images and hasattr(candidate, 'content'):
                # Some responses might have a different structure
                logger.warning("No inline_data found, trying alternative extraction")
                # Log the response structure for debugging
                logger.debug(f"Response structure: {dir(candidate.content)}")
            
            if not images:
                raise ValueError("Failed to extract image from response")
            
            logger.info(f"Generated {len(images)} images")
            return images

        except Exception as e:
            logger.error(f"Image generation failed: {e}", exc_info=True)
            raise

    async def edit_image(
        self,
        base_image: bytes,
        edit_prompt: str,
        mime_type: str = "image/jpeg"
    ) -> List[bytes]:
        """Edit an existing image using conversational editing.
        
        Args:
            base_image: Original image bytes
            edit_prompt: Editing instruction
            mime_type: Image MIME type
            
        Returns:
            List of edited image bytes
        """
        try:
            # Create multimodal prompt with image + edit instruction
            image_part = types.Part.from_bytes(
                data=base_image,
                mime_type=mime_type
            )
            
            config = types.GenerateContentConfig(
                response_modalities=["IMAGE"]
            )
            
            response = self.client.models.generate_content(
                model=self.generation_model,
                contents=[image_part, edit_prompt],
                config=config
            )
            
            # Extract edited images
            images = []
            if response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            data = part.inline_data.data
                            if isinstance(data, str):
                                image_bytes = base64.b64decode(data)
                            else:
                                image_bytes = data
                            images.append(image_bytes)
            
            if not images:
                raise ValueError("Failed to extract edited image from response")
            
            logger.info(f"Generated {len(images)} edited images")
            return images

        except Exception as e:
            logger.error(f"Image editing failed: {e}")
            raise

    async def analyze_and_generate(
        self,
        reference_image: Optional[bytes],
        task: str
    ) -> Dict[str, Union[str, List[bytes]]]:
        """Combined workflow: analyze a reference image, then generate based on it.
        
        Args:
            reference_image: Optional reference image
            task: Combined task description
            
        Returns:
            Dict with 'analysis' (str) and 'generated_images' (List[bytes])
        """
        try:
            result = {
                'analysis': None,
                'generated_images': []
            }
            
            # Step 1: Analyze reference if provided
            if reference_image:
                analysis = await self.analyze_image(
                    reference_image,
                    prompt=f"Analyze this image for the following task: {task}"
                )
                result['analysis'] = analysis
                
                # Step 2: Generate based on analysis
                generation_prompt = f"Based on this analysis: {analysis}\n\nTask: {task}"
            else:
                generation_prompt = task
            
            # Generate images
            images = await self.generate_image(generation_prompt)
            result['generated_images'] = images
            
            return result

        except Exception as e:
            logger.error(f"Combined analyze+generate workflow failed: {e}")
            raise


# Utility functions for Discord/Telegram adapters
def download_image_from_url(url: str) -> bytes:
    """Download image from URL and return bytes."""
    import httpx
    response = httpx.get(url)
    response.raise_for_status()
    return response.content


def save_image(image_bytes: bytes, output_path: Union[str, Path]) -> None:
    """Save image bytes to disk."""
    with open(output_path, 'wb') as f:
        f.write(image_bytes)
    logger.info(f"Saved image to {output_path}")


def image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 data URI."""
    b64 = base64.b64encode(image_bytes).decode()
    return f"data:image/jpeg;base64,{b64}"