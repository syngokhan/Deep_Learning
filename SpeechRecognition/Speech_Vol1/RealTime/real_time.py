import pyaudio
import websockets
import asyncio
import base64
import json
import openai

speech_api= "key_1"
openai_api = "key_2"

#Open AI
openai.api_key = openai_api

def ask_computer(prompt):
    response = openai.Completion.create(       
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens = 100)
    return response["choices"][0]["text"]



FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS  = 1
RATE = 16000

p = pyaudio.PyAudio()

stream = p.open(
    format = FORMAT,
    channels=CHANNELS,
    rate = RATE,
    input = True,
    frames_per_buffer= FRAMES_PER_BUFFER
)

URL = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"

async def send_receice():
    print(f'Connecting websocket to url ${URL}')
    async with websockets.connect(
        URL,
        ping_timeout = 20,
        ping_interval = 5,
        extra_headers = {"Authorization" : speech_api}
    ) as _ws:
        await asyncio.sleep(0.1)
        print("Receiving SessionBegins ...")
        session_begins = await _ws.recv()
        print(session_begins)
        print("Sending Messages")

        async def send():
            while True:
                try:
                    data = stream.read(FRAMES_PER_BUFFER,exception_on_overflow=False)
                    data = base64.b64encode(data).decode("utf-8")
                    json_data = json.dumps({"audio_data":str(data)})
                    await _ws.send(json_data)
                except websockets.exceptions.ConnectionClosedError as e:
                    print(e)
                    assert e.code == 4008
                    break
                except Exception as e:
                    assert False, "Not a websocket 4008 error"
                await asyncio.sleep(0.01)

            return True

        async def receive():
            while True:
                try:
                    result_str = await _ws.recv()
                    result = json.loads(result_str)
                    prompt = result["text"]
                    if prompt and result["message_type"] == "FinalTranscript":
                        
                        print("Me : ",prompt)
                        response = ask_computer(prompt)
                        print("Bot : " , response)

                except websockets.exceptions.ConnectionClosedError as e:
                    print(e)
                    assert e.code == 4008
                    break

                except Exception as e:
                    assert False, "Not a websocket 4008 error"


        send_result,receive_result = await asyncio.gather(send(),receive())

asyncio.run(send_receice())
    