"""Discord adapter with image support.

Extends the base Discord adapter to handle image attachments
for vision analysis and generation tasks.
"""
import asyncio
import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import discord
from discord.ext import commands

logger = logging.getLogger(__name__)


class VisionDiscordAdapter:
    """Discord bot adapter with vision capabilities."""
    
    def __init__(self, token: str, router, vision_provider):
        """Initialize Discord adapter with vision support.
        
        Args:
            token: Discord bot token
            router: Message router instance
            vision_provider: VisionProvider instance
        """
        self.token = token
        self.router = router
        self.vision = vision_provider
        
        # Configure intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        
        self.bot = commands.Bot(command_prefix="!", intents=intents)
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up Discord event handlers."""
        
        @self.bot.event
        async def on_ready():
            logger.info(f"Discord bot ready: {self.bot.user}")
        
        @self.bot.event
        async def on_message(message: discord.Message):
            # Ignore own messages
            if message.author == self.bot.user:
                return
            
            # Only respond to DMs or @mentions
            is_dm = isinstance(message.channel, discord.DMChannel)
            is_mentioned = self.bot.user in message.mentions
            
            if not (is_dm or is_mentioned):
                return
            
            # Extract message content
            content = message.content
            if is_mentioned:
                # Remove bot mention from content
                content = content.replace(f"<@{self.bot.user.id}>", "").strip()
            
            # Handle images
            images = await self._extract_images(message)
            
            try:
                # Route to orchestrator with image context
                response = await self._handle_message_with_images(
                    user_id=f"discord:{message.author.id}",
                    content=content,
                    images=images
                )
                
                # Send response (may include generated images)
                await self._send_response(message.channel, response)
                
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                await message.channel.send("Sorry, I encountered an error processing your request.")
    
    async def _extract_images(self, message: discord.Message) -> List[dict]:
        """Extract images from Discord message attachments.
        
        Args:
            message: Discord message
            
        Returns:
            List of dicts with 'data' (bytes), 'mime_type', and 'filename'
        """
        images = []
        
        for attachment in message.attachments:
            # Check if attachment is an image
            if attachment.content_type and attachment.content_type.startswith('image/'):
                try:
                    # Download image
                    image_bytes = await attachment.read()
                    
                    images.append({
                        'data': image_bytes,
                        'mime_type': attachment.content_type,
                        'filename': attachment.filename
                    })
                    
                    logger.info(f"Extracted image: {attachment.filename} ({len(image_bytes)} bytes)")
                    
                except Exception as e:
                    logger.error(f"Failed to download attachment {attachment.filename}: {e}")
        
        return images
    
    async def _handle_message_with_images(
        self,
        user_id: str,
        content: str,
        images: List[dict]
    ) -> dict:
        """Handle message with optional images.
        
        Args:
            user_id: Discord user ID
            content: Message text
            images: List of image dicts
            
        Returns:
            Response dict with 'text' and optional 'images' (List[bytes])
        """
        # Determine task type
        is_generation = self._is_generation_request(content)
        is_analysis = len(images) > 0 or self._is_analysis_request(content)
        
        if is_analysis and images:
            # Image analysis workflow
            return await self._handle_image_analysis(user_id, content, images)
        
        elif is_generation:
            # Image generation workflow
            return await self._handle_image_generation(user_id, content)
        
        else:
            # Standard text-only workflow
            response_text = await self.router.route_message(user_id, content)
            return {'text': response_text, 'images': []}
    
    def _is_generation_request(self, content: str) -> bool:
        """Detect if message is requesting image generation."""
        triggers = [
            'generate an image',
            'create an image',
            'draw me',
            'make me a picture',
            'generate a picture',
            'create a visual',
            'show me what',
            'visualize',
        ]
        content_lower = content.lower()
        return any(trigger in content_lower for trigger in triggers)
    
    def _is_analysis_request(self, content: str) -> bool:
        """Detect if message is requesting image analysis."""
        triggers = [
            'what is in this image',
            'describe this',
            'what do you see',
            'analyze this',
            'read this',
            'what does this say',
            'ocr',
        ]
        content_lower = content.lower()
        return any(trigger in content_lower for trigger in triggers)
    
    async def _handle_image_analysis(
        self,
        user_id: str,
        content: str,
        images: List[dict]
    ) -> dict:
        """Handle image analysis request.
        
        Args:
            user_id: User ID
            content: Message text
            images: Image attachments
            
        Returns:
            Response dict
        """
        try:
            if len(images) == 1:
                # Single image analysis
                analysis = await self.vision.analyze_image(
                    image_data=images[0]['data'],
                    prompt=content or "Describe this image in detail.",
                    mime_type=images[0]['mime_type']
                )
            else:
                # Multiple image analysis
                analysis = await self.vision.analyze_multiple_images(
                    images=images,
                    prompt=content or "Describe these images and how they relate to each other."
                )
            
            return {'text': analysis, 'images': []}
        
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {
                'text': f"Sorry, I couldn't analyze the image(s): {str(e)}",
                'images': []
            }
    
    async def _handle_image_generation(
        self,
        user_id: str,
        content: str
    ) -> dict:
        """Handle image generation request.
        
        Args:
            user_id: User ID
            content: Generation prompt
            
        Returns:
            Response dict with generated images
        """
        try:
            # Extract aspect ratio if specified
            aspect_ratio = "1:1"  # Default
            if "landscape" in content.lower() or "wide" in content.lower():
                aspect_ratio = "16:9"
            elif "portrait" in content.lower() or "tall" in content.lower():
                aspect_ratio = "9:16"
            
            # Generate images
            generated = await self.vision.generate_image(
                prompt=content,
                num_images=1,
                aspect_ratio=aspect_ratio
            )
            
            return {
                'text': f"Generated image based on: {content}",
                'images': generated
            }
        
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return {
                'text': f"Sorry, I couldn't generate the image: {str(e)}",
                'images': []
            }
    
    async def _send_response(
        self,
        channel: discord.TextChannel,
        response: dict
    ):
        """Send response to Discord channel.
        
        Args:
            channel: Discord channel
            response: Response dict with 'text' and optional 'images'
        """
        # Send text response
        if response['text']:
            # Split long messages
            chunks = self._split_message(response['text'], 2000)
            for chunk in chunks:
                await channel.send(chunk)
        
        # Send generated images
        if response.get('images'):
            for idx, image_bytes in enumerate(response['images']):
                # Convert to Discord file
                file = discord.File(
                    fp=BytesIO(image_bytes),
                    filename=f"generated_{idx}.png"
                )
                await channel.send(file=file)
    
    def _split_message(self, text: str, max_length: int = 2000) -> List[str]:
        """Split long messages into chunks."""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        while text:
            if len(text) <= max_length:
                chunks.append(text)
                break
            
            # Find last newline before max_length
            split_at = text.rfind('\n', 0, max_length)
            if split_at == -1:
                split_at = max_length
            
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip()
        
        return chunks
    
    async def start(self):
        """Start the Discord bot."""
        logger.info("Starting Discord bot...")
        await self.bot.start(self.token)
    
    async def stop(self):
        """Stop the Discord bot."""
        logger.info("Stopping Discord bot...")
        await self.bot.close()


# Example integration with your existing router
async def create_vision_discord_adapter(config, router, vision_provider):
    """Factory function to create vision-enabled Discord adapter.
    
    Args:
        config: Config object with discord token
        router: Router instance
        vision_provider: VisionProvider instance
        
    Returns:
        VisionDiscordAdapter instance
    """
    token = config.get('discord_bot_token')
    if not token:
        raise ValueError("Discord bot token not configured")
    
    adapter = VisionDiscordAdapter(token, router, vision_provider)
    return adapter