"""
Deepgram Speech-to-Text (Fixed for current SDK)
Works with deepgram-sdk version 3.x
"""

import asyncio
import json


class DeepgramSTT:
    """Deepgram real-time speech-to-text"""
    
    def __init__(self, api_key, callback=None):
        """
        Initialize Deepgram STT
        
        Args:
            api_key: Deepgram API key
            callback: Function to call with transcribed text
        """
        self.api_key = api_key
        self.callback = callback
        self.dg_connection = None
        self.microphone = None
        self.is_listening = False
        
        # Initialize client - works with both old and new SDK
        try:
            from deepgram import (
                DeepgramClient,
                DeepgramClientOptions,
                LiveTranscriptionEvents,
                LiveOptions,
                Microphone,
            )
            self.has_new_api = True
            config = DeepgramClientOptions(options={"keepalive": "true"})
            self.deepgram = DeepgramClient(api_key, config)
        except ImportError:
            # Fallback for older SDK
            from deepgram import Deepgram
            self.has_new_api = False
            self.deepgram = Deepgram(api_key)
        
        print("✓ Deepgram STT initialized")
    
    async def start_listening(self):
        """Start listening to microphone and transcribing"""
        try:
            if self.has_new_api:
                await self._start_listening_new()
            else:
                await self._start_listening_old()
        except Exception as e:
            print(f"❌ Error starting Deepgram STT: {e}")
            raise
    
    async def _start_listening_new(self):
        """New SDK API (3.x+)"""
        from deepgram import LiveTranscriptionEvents, LiveOptions, Microphone
        
        # Create connection
        self.dg_connection = self.deepgram.listen.asyncwebsocket.v("1")
        
        # Register event handlers
        self.dg_connection.on(LiveTranscriptionEvents.Transcript, self._on_message)
        self.dg_connection.on(LiveTranscriptionEvents.Error, self._on_error)
        
        # Configure options
        options = LiveOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
            interim_results=True,
            utterance_end_ms="1000",
            vad_events=True,
            endpointing=300
        )
        
        # Start connection
        await self.dg_connection.start(options)
        
        # Start microphone
        self.microphone = Microphone(self.dg_connection.send)
        self.microphone.start()
        
        self.is_listening = True
        print("🎤 Listening... (speak now)")
    
    async def _start_listening_old(self):
        """Old SDK API (2.x)"""
        # Setup connection with old API
        options = {
            'model': 'nova-2',
            'language': 'en-US',
            'smart_format': True,
            'interim_results': True,
            'utterance_end_ms': 1000,
            'vad_events': True,
            'endpointing': 300
        }
        
        self.dg_connection = await self.deepgram.transcription.live(options)
        self.dg_connection.registerHandler(
            self.dg_connection.event.TRANSCRIPT_RECEIVED,
            self._on_message_old
        )
        
        # Start microphone
        try:
            from deepgram import Microphone
            self.microphone = Microphone(self.dg_connection.send)
            self.microphone.start()
        except:
            print("⚠ Microphone not available in old SDK")
        
        self.is_listening = True
        print("🎤 Listening... (speak now)")
    
    async def stop_listening(self):
        """Stop listening"""
        self.is_listening = False
        
        if self.microphone:
            self.microphone.finish()
            self.microphone = None
        
        if self.dg_connection:
            if self.has_new_api:
                await self.dg_connection.finish()
            else:
                await self.dg_connection.finish()
            self.dg_connection = None
        
        print("🛑 Stopped listening")
    
    def _on_message(self, result, **kwargs):
        """Handle transcription results (new SDK)"""
        try:
            sentence = result.channel.alternatives[0].transcript
            
            if len(sentence) > 0:
                if result.is_final:
                    print(f"📝 Final: {sentence}")
                    if self.callback:
                        self.callback(sentence)
                else:
                    print(f"⏳ Interim: {sentence}", end='\r')
        except Exception as e:
            print(f"❌ Error processing message: {e}")
    
    def _on_message_old(self, result, **kwargs):
        """Handle transcription results (old SDK)"""
        try:
            transcript = result.get('channel', {}).get('alternatives', [{}])[0].get('transcript', '')
            is_final = result.get('is_final', False)
            
            if len(transcript) > 0:
                if is_final:
                    print(f"📝 Final: {transcript}")
                    if self.callback:
                        self.callback(transcript)
                else:
                    print(f"⏳ Interim: {transcript}", end='\r')
        except Exception as e:
            print(f"❌ Error processing message: {e}")
    
    def _on_error(self, error, **kwargs):
        """Handle errors"""
        print(f"❌ Deepgram error: {error}")


# Example usage
async def example_callback(text):
    """Example callback function"""
    print(f"\n✅ You said: {text}")


async def main():
    """Test Deepgram STT"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("DEEPGRAM_API_KEY")
    
    if not api_key:
        print("❌ Please set DEEPGRAM_API_KEY in .env file")
        return
    
    stt = DeepgramSTT(api_key, callback=example_callback)
    
    try:
        await stt.start_listening()
        
        # Listen for 30 seconds
        print("Speak for 30 seconds...")
        await asyncio.sleep(30)
        
        await stt.stop_listening()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
        await stt.stop_listening()


if __name__ == "__main__":
    asyncio.run(main())