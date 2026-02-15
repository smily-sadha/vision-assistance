"""
Deepgram Speech-to-Text (Fixed for current SDK)
Works with deepgram-sdk version 3.x
"""

import asyncio
import json

try:
    import speech_recognition as sr
except Exception:
    sr = None


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
        # Interim-to-final handling: when interim stops updating for this many ms,
        # treat last interim as final and invoke the callback.
        self._last_interim = None
        self._interim_task = None
        self._interim_timeout_ms = 500
        self._recv_task = None
        self._local_stt_task = None
        
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

        # Create connection with robust fallbacks for different SDK releases
        listen_obj = getattr(self.deepgram, 'listen', None)
        if listen_obj is None:
            raise RuntimeError("Deepgram client has no 'listen' attribute - incompatible SDK")

        self.dg_connection = None

        # Try common attribute names used across SDK versions
        ws_factory = None
        for name in ('asyncwebsocket', 'websocket', 'async_websocket', 'websocket_async'):
            if hasattr(listen_obj, name):
                ws_factory = getattr(listen_obj, name)
                break

        try:
            if ws_factory is None:
                # If listen_obj itself is callable, try calling it
                if callable(listen_obj):
                    candidate = listen_obj()
                else:
                    candidate = listen_obj
            else:
                candidate = ws_factory() if callable(ws_factory) else ws_factory

            # Many variants expose a `v()` helper to pick protocol version
            if hasattr(candidate, 'v'):
                self.dg_connection = candidate.v("1")
            else:
                self.dg_connection = candidate
        except Exception as e:
            print(f"⚠ Failed to create websocket via listen attributes: {e}")
            # Fallback: try transcription.live() (older async path on some clients)
            try:
                self.dg_connection = await self.deepgram.transcription.live()
            except Exception as e2:
                raise RuntimeError("Unable to create Deepgram live connection") from e2

        # Register event handlers (try multiple APIs)
        try:
            self.dg_connection.on(LiveTranscriptionEvents.Transcript, self._on_message)
            self.dg_connection.on(LiveTranscriptionEvents.Error, self._on_error)
        except Exception:
            try:
                self.dg_connection.registerHandler(
                    getattr(self.dg_connection, 'event', None).TRANSCRIPT_RECEIVED,
                    self._on_message,
                )
            except Exception:
                # If registration fails, continue; handler may still be invoked differently
                print("⚠ Could not register event handlers using known methods")
                # we'll attempt a generic receive loop later

        # Configure options
        options = LiveOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
            interim_results=True,
            utterance_end_ms=500,
            vad_events=True,
            endpointing=200
        )

        # Start connection (try common start/connect variations)
        try:
            # preferred async start
            await self.dg_connection.start(options)
        except Exception:
            # try connect or start without await
            if hasattr(self.dg_connection, 'connect'):
                conn = self.dg_connection.connect(options)
                if asyncio.iscoroutine(conn):
                    await conn
            elif hasattr(self.dg_connection, 'start') and not asyncio.iscoroutinefunction(self.dg_connection.start):
                try:
                    self.dg_connection.start(options)
                except Exception:
                    pass

        # Start microphone if available
        try:
            self.microphone = Microphone(self.dg_connection.send)
            self.microphone.start()
        except Exception:
            print("⚠ Microphone not available in new SDK path")
            # Fallback microphone implementation using sounddevice (optional dependency)
            try:
                self.microphone = _MicrophoneFallback(self.dg_connection.send)
                self.microphone.start()
            except Exception as e:
                print(f"⚠ Microphone fallback failed: {e}")

        # If we couldn't register event handlers earlier, try a generic receive loop
        if not hasattr(self.dg_connection, 'on') or 'Could not register event handlers' in "":
            # try to create a receive loop if dg_connection is an async iterator or has recv/receive
            try:
                if hasattr(self.dg_connection, '__aiter__') or hasattr(self.dg_connection, 'receive') or hasattr(self.dg_connection, 'recv'):
                    self._recv_task = asyncio.get_running_loop().create_task(self._recv_loop())
            except Exception:
                pass

        # If DG connection doesn't expose send/on handlers, start local STT fallback
        if (not hasattr(self.dg_connection, 'on') or not hasattr(self.dg_connection, 'send')) and sr is not None:
            try:
                print("⚠ Using local speech_recognition fallback for STT")
                self._local_stt_task = asyncio.get_running_loop().create_task(self._local_stt_loop())
            except Exception:
                pass

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

        if self._recv_task:
            try:
                self._recv_task.cancel()
            except Exception:
                pass
            self._recv_task = None
        if self._local_stt_task:
            try:
                self._local_stt_task.cancel()
            except Exception:
                pass
            self._local_stt_task = None
        
        print("🛑 Stopped listening")
    
    def _on_message(self, result, **kwargs):
        """Handle transcription results (new SDK)"""
        try:
            sentence = result.channel.alternatives[0].transcript
            
            if len(sentence) > 0:
                if result.is_final:
                    # Cancel any pending interim-finalizer and emit final
                    if self._interim_task:
                        try:
                            self._interim_task.cancel()
                        except Exception:
                            pass
                        self._interim_task = None
                        self._last_interim = None

                    print(f"📝 Final: {sentence}")
                    if self.callback:
                        self.callback(sentence)
                else:
                    # Store interim and schedule a timer to finalize on short pause
                    self._last_interim = sentence
                    print(f"⏳ Interim: {sentence}", end='\r')
                    try:
                        if self._interim_task:
                            try:
                                self._interim_task.cancel()
                            except Exception:
                                pass
                        self._interim_task = asyncio.get_running_loop().create_task(
                            self._delayed_finalize_interim(sentence, self._interim_timeout_ms)
                        )
                    except Exception:
                        # If no running loop, ignore (best-effort)
                        pass
        except Exception as e:
            print(f"❌ Error processing message: {e}")
    
    def _on_message_old(self, result, **kwargs):
        """Handle transcription results (old SDK)"""
        try:
            transcript = result.get('channel', {}).get('alternatives', [{}])[0].get('transcript', '')
            is_final = result.get('is_final', False)
            
            if len(transcript) > 0:
                if is_final:
                    if self._interim_task:
                        try:
                            self._interim_task.cancel()
                        except Exception:
                            pass
                        self._interim_task = None
                        self._last_interim = None

                    print(f"📝 Final: {transcript}")
                    if self.callback:
                        self.callback(transcript)
                else:
                    self._last_interim = transcript
                    print(f"⏳ Interim: {transcript}", end='\r')
                    try:
                        if self._interim_task:
                            try:
                                self._interim_task.cancel()
                            except Exception:
                                pass
                        # old SDK handlers may be called from non-async context, still schedule
                        self._interim_task = asyncio.get_running_loop().create_task(
                            self._delayed_finalize_interim(transcript, self._interim_timeout_ms)
                        )
                    except Exception:
                        pass
        except Exception as e:
            print(f"❌ Error processing message: {e}")
    
    def _on_error(self, error, **kwargs):
        """Handle errors"""
        print(f"❌ Deepgram error: {error}")

    async def _delayed_finalize_interim(self, text, timeout_ms):
        """Wait `timeout_ms` milliseconds; if interim unchanged, treat as final."""
        try:
            await asyncio.sleep(timeout_ms / 1000.0)
            if self._last_interim == text:
                # clear state and emit as final
                self._interim_task = None
                self._last_interim = None
                print(f"📝 Final (on pause): {text}")
                if self.callback:
                    try:
                        self.callback(text)
                    except Exception as e:
                        print(f"❌ Error invoking callback: {e}")
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"❌ Error in interim finalizer: {e}")

    async def _recv_loop(self):
        """Generic receive loop fallback for dg_connection variants.
        Iterates/awaits messages and dispatches to message handlers.
        """
        try:
            conn = self.dg_connection
            # prefer async iterator
            if hasattr(conn, '__aiter__'):
                async for msg in conn:
                    try:
                        # Try to treat like new SDK message
                        if hasattr(msg, 'channel'):
                            self._on_message(msg)
                        else:
                            # assume old-style dict
                            self._on_message_old(msg)
                    except Exception:
                        pass
            else:
                # Try generic receive() or recv()
                recv = getattr(conn, 'receive', None) or getattr(conn, 'recv', None)
                if recv is None:
                    return
                while True:
                    msg = recv()
                    if asyncio.iscoroutine(msg):
                        msg = await msg
                    try:
                        if hasattr(msg, 'channel'):
                            self._on_message(msg)
                        else:
                            self._on_message_old(msg)
                    except Exception:
                        pass
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"⚠ Receive loop error: {e}")

    async def _local_stt_loop(self):
        """Local STT fallback using `speech_recognition` (runs in executor)."""
        if sr is None:
            print("⚠ speech_recognition not installed; local STT unavailable")
            return

        recognizer = sr.Recognizer()
        loop = asyncio.get_running_loop()

        def capture_and_recognize():
            # Try speech_recognition Microphone (PyAudio) first
            try:
                with sr.Microphone() as source:
                    audio = recognizer.listen(source, phrase_time_limit=5)
                    try:
                        return recognizer.recognize_google(audio)
                    except Exception:
                        return None
            except Exception:
                # Fallback to sounddevice if available
                try:
                    import sounddevice as sd
                    import numpy as np
                    duration = 4.5
                    fs = 16000
                    rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
                    sd.wait()
                    arr = np.asarray(rec, dtype='int16')
                    b = arr.tobytes()
                    audio_data = sr.AudioData(b, fs, 2)
                    try:
                        return recognizer.recognize_google(audio_data)
                    except Exception:
                        return None
                except Exception:
                    return None

        try:
            while self.is_listening:
                text = await loop.run_in_executor(None, capture_and_recognize)
                if text:
                    print(f"📝 Local STT: {text}")
                    if self.callback:
                        try:
                            self.callback(text)
                        except Exception as e:
                            print(f"❌ Error invoking callback in local STT: {e}")
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"⚠ Local STT loop error: {e}")


class _MicrophoneFallback:
    """Fallback microphone that uses sounddevice to capture audio and send bytes.
    This is optional and only used when the Deepgram SDK Microphone helper isn't available.
    """
    def __init__(self, send_fn, sample_rate=16000, blocksize=1024):
        self._send = send_fn
        self._stream = None
        self._sample_rate = sample_rate
        self._blocksize = blocksize

    def start(self):
        try:
            import sounddevice as sd
        except Exception:
            raise RuntimeError('sounddevice not available for microphone fallback')

        def callback(indata, frames, time, status):
            try:
                # indata is float32 or int16 depending on dtype; convert to int16 bytes
                if indata.dtype.kind == 'f':
                    import numpy as np
                    arr = (indata * 32767).astype('int16')
                else:
                    arr = indata
                # mono -> bytes
                b = arr.tobytes()
                try:
                    res = self._send(b)
                    # if send is coroutine, schedule it
                    if asyncio.iscoroutine(res):
                        asyncio.get_running_loop().create_task(res)
                except Exception:
                    pass
            except Exception:
                pass

        self._stream = sd.InputStream(samplerate=self._sample_rate, channels=1, dtype='int16', blocksize=self._blocksize, callback=callback)
        self._stream.start()

    def finish(self):
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
        except Exception:
            pass


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