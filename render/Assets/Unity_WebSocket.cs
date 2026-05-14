using System.Collections;
using UnityEngine;
using NativeWebSocket;  // ← 설치한 패키지

public class Unity_WebSocket : MonoBehaviour
{
    WebSocket ws;
    public string url = "ws://127.0.0.1:8000/ws";  // ws:// 로 바뀜

    async void Start()
    {
        ws = new WebSocket(url);

        // FastAPI로부터 메시지 수신
        ws.OnMessage += (bytes) => {
            string data = System.Text.Encoding.UTF8.GetString(bytes);
            Debug.Log("수신: " + data);
        };

        ws.OnOpen += () => Debug.Log("연결됨");
        ws.OnClose += (e) => Debug.Log("연결 끊김");
        ws.OnError += (e) => Debug.LogError("에러: " + e);

        await ws.Connect();  // 연결
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        ws.DispatchMessageQueue();  // 매 프레임 메시지 수신 처리 (필수!)
#endif
    }

    // 데이터 전송 함수
    public async void SendScore(string name, int score)
    {
        if (ws.State == WebSocketState.Open)
        {
            string json = $"{{\"name\":\"{name}\",\"score\":{score}}}";
            await ws.SendText(json);
        }
    }

    // 앱 종료 시 연결 해제
    async void OnDestroy()
    {
        await ws.Close();
    }
}
