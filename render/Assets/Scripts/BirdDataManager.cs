using System;
using System.Collections.Generic;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.VFX;

public class BirdDataManager : MonoBehaviour
{
    [System.Serializable]
    public class LatLonPoint { public float lat; public float lon; public float altitude_m; }

    [System.Serializable]
    public class EnvironmentData { public float tailwind; }

    [System.Serializable]
    public class AttributionData { public float ws_925; }

    [System.Serializable]
    public class FrameMessage
    {
        public string message_type;
        public LatLonPoint position; 
        public EnvironmentData environment;
        public AttributionData attributions;
        public float weather_key; 
    }

    [Header("[1] FASTAPI NETWORK SETTINGS")]
    public string serverIP = "127.0.0.1";
    public string serverPort = "8080";
    public string wsEndpoint = "ws";

    [Header("[2] REFERENCES (오브젝트 결속)")]
    public VisualEffect flockVFX;
    public Transform leaderCoreTransform;
    public Transform gridBoxTransform; 
    public Transform mainCameraTransform;

    [Header("💡 [3] BIRD FREEDOM SETTINGS (공간 스케일 배율)")]
    public float birdSmoothTime = 0.4f;
    public float birdMovementMultiplier = 4.0f;
    public float maxSpeed = 70f;

    [Header("[4] FIXED COMPASS LAYERED DELAY")]
    public float gridPositionSmoothTime = 1.5f;
    public float cameraFollowSmoothTime = 2.5f;

    [Header("[5] LIVE MONITOR (실시간 전광판 계측기)")]
    [TextArea(2, 4)]
    [SerializeField] private string rawServerDataLog = "No Data Yet";
    [SerializeField] private Vector3 liveTargetPos;
    [SerializeField] private Vector3 liveCurrentPos;

    [Header("XAI 기여도 실시간 관측")]
    public float liveTailwind;
    public float liveHeadwind;
    public float liveWeatherKey;

    private float minLat = -5.7379746f;
    private float maxLat = 43.662685f;
    private float minLon = -76.26561f;
    private float maxLon = -48.195133f;
    private float minAlt = -47.35702f;
    private float maxAlt = 235.4282f;

    private Vector3 birdVelocity = Vector3.zero;
    private Vector3 gridVelocity = Vector3.zero;
    private Vector3 cameraVelocity = Vector3.zero; 
    private Vector3 cameraInitialOffset; 

    private ClientWebSocket webSocket = null;
    private CancellationTokenSource cts;
    private Queue<string> rawPacketQueue = new Queue<string>(); 
    private readonly object queueLock = new object();

    void Start()
    {
        Debug.Log("<color=lime><b>[동기화 통합 마스터]</b> 서버 매핑 엇박자 가드 엔진 가동.</color>");

        Vector3 startPos = Vector3.zero; 
        liveTargetPos = startPos;
        liveCurrentPos = startPos;

        if (leaderCoreTransform != null) leaderCoreTransform.position = startPos;
        if (flockVFX == null) flockVFX = FindObjectOfType<VisualEffect>();

        if (gridBoxTransform != null) gridBoxTransform.rotation = Quaternion.identity;

        if (mainCameraTransform != null && gridBoxTransform != null)
        {
            cameraInitialOffset = mainCameraTransform.position - gridBoxTransform.position;
        }

        string connectionUrl = $"ws://{serverIP}:{serverPort}/{wsEndpoint}";
        if (connectionUrl.Contains("up.railway.app")) connectionUrl = connectionUrl.Replace("ws://", "wss://");

        cts = new CancellationTokenSource();
        Task.Run(() => ConnectAndReceiveLoop(connectionUrl, cts.Token));
    }

    void Update()
    {
        bool hasNewData = false;

        while (rawPacketQueue.Count > 0)
        {
            string rawJsonPacket = null;
            lock (queueLock) 
            { 
                if (rawPacketQueue.Count > 0) rawJsonPacket = rawPacketQueue.Dequeue(); 
            }

            if (!string.IsNullOrEmpty(rawJsonPacket))
            {
                rawServerDataLog = rawJsonPacket;
                FrameMessage parsed = null;
                try { parsed = JsonUtility.FromJson<FrameMessage>(rawJsonPacket); } catch { }

                if (parsed != null && parsed.position != null)
                {
                    float actualLat = parsed.position.lat;
                    float actualLon = parsed.position.lon;
                    float actualAlt = parsed.position.altitude_m;

                    if (actualLat < minLat) minLat = actualLat; if (actualLat > maxLat) maxLat = actualLat;
                    if (actualLon < minLon) minLon = actualLon; if (actualLon > maxLon) maxLon = actualLon;
                    if (actualAlt < minAlt) minAlt = actualAlt; if (actualAlt > maxAlt) maxAlt = actualAlt;

                    liveTailwind = (parsed.environment != null) ? parsed.environment.tailwind : 0f;
                    liveHeadwind = (parsed.attributions != null) ? parsed.attributions.ws_925 : 0f;
                    liveWeatherKey = parsed.weather_key;

                    if (liveTailwind == 0f) liveTailwind = ExtractFloatFromJson(rawJsonPacket, "tailwind");
                    if (liveHeadwind == 0f) 
                    {
                        liveHeadwind = ExtractFloatFromJson(rawJsonPacket, "ws_925");
                        if (liveHeadwind == 0f) liveHeadwind = ExtractFloatFromJson(rawJsonPacket, "headwind");
                    }
                    if (liveWeatherKey == 0f) liveWeatherKey = ExtractFloatFromJson(rawJsonPacket, "weather_key");

                    float normX = (maxLon - minLon != 0) ? (actualLon - minLon) / (maxLon - minLon) : 0.5f;
                    float normY = (maxAlt - minAlt != 0) ? (actualAlt - minAlt) / (maxAlt - minAlt) : 0.5f;
                    float normZ = (maxLat - minLat != 0) ? (actualLat - minLat) / (maxLat - minLat) : 0.5f;

                    float boxWidth = 100f; float boxHeight = 50f; float boxLength = 100f;

                    float finalX = (normX - 0.5f) * boxWidth * birdMovementMultiplier; 
                    float finalY = (normY - 0.5f) * boxHeight * birdMovementMultiplier;
                    float finalZ = (normZ - 0.5f) * boxLength * birdMovementMultiplier;

                    Vector3 nextTarget = new Vector3(finalX, finalY, finalZ);
                    
                    if (Vector3.Distance(liveTargetPos, nextTarget) > 0.001f)
                    {
                        hasNewData = true;
                    }
                    liveTargetPos = nextTarget;
                }
            }
        }

        if (!hasNewData)
        {
            float fakeSwayX = Mathf.Sin(Time.time * 0.7f) * 1.2f;
            float fakeSwayZ = Mathf.Cos(Time.time * 0.5f) * 1.2f;
            liveTargetPos += new Vector3(fakeSwayX, 0f, fakeSwayZ);
        }

        liveCurrentPos = Vector3.SmoothDamp(liveCurrentPos, liveTargetPos, ref birdVelocity, birdSmoothTime, maxSpeed, Time.deltaTime);

        if (gridBoxTransform != null)
        {
            gridBoxTransform.position = Vector3.SmoothDamp(gridBoxTransform.position, liveCurrentPos, ref gridVelocity, gridPositionSmoothTime, maxSpeed, Time.deltaTime);
            gridBoxTransform.rotation = Quaternion.identity; 
        }

        if (mainCameraTransform != null && gridBoxTransform != null)
        {
            Vector3 targetCameraPos = gridBoxTransform.position + cameraInitialOffset;
            mainCameraTransform.position = Vector3.SmoothDamp(mainCameraTransform.position, targetCameraPos, ref cameraVelocity, cameraFollowSmoothTime, maxSpeed, Time.deltaTime);
        }

        float swayIntensity = 1.8f; float swaySpeed = 1.2f;     
        Vector3 noiseSway = new Vector3(Mathf.PerlinNoise(Time.time * swaySpeed, 0f) - 0.5f, Mathf.PerlinNoise(0f, Time.time * swaySpeed) - 0.5f, Mathf.PerlinNoise(Time.time * swaySpeed, Time.time * swaySpeed) - 0.5f) * swayIntensity;
        Vector3 finalVisualPos = liveCurrentPos + noiseSway;

        if (leaderCoreTransform != null)
        {
            leaderCoreTransform.position = finalVisualPos;
            Vector3 moveDirection = finalVisualPos - (liveCurrentPos - birdVelocity * Time.deltaTime);
            if (moveDirection.magnitude > 0.001f)
            {
                leaderCoreTransform.forward = moveDirection.normalized;
            }
        }

        if (flockVFX != null)
        {
            flockVFX.SetVector3("LeaderPosition", finalVisualPos);
            flockVFX.SetFloat("TailwindIntensity", liveTailwind);
            flockVFX.SetFloat("HeadwindIntensity", liveHeadwind);
            flockVFX.SetFloat("WeatherIntensity", liveWeatherKey);
        }
    }

    private float ExtractFloatFromJson(string json, string key)
    {
        try
        {
            if (!json.Contains(key)) return 0f;
            int index = json.IndexOf("\"" + key + "\"");
            if (index == -1) index = json.IndexOf(key);
            int colonIndex = json.IndexOf(":", index);
            int commaIndex = json.IndexOf(",", colonIndex);
            if (commaIndex == -1) commaIndex = json.IndexOf("}", colonIndex);
            string valueStr = json.Substring(colonIndex + 1, commaIndex - colonIndex - 1).Replace("\"", "").Trim();
            float result;
            if (float.TryParse(valueStr, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out result)) return result;
        }
        catch { }
        return 0f;
    }

    private async Task ConnectAndReceiveLoop(string url, CancellationToken token)
    {
        try
        {
            webSocket = new ClientWebSocket(); Uri serverUri = new Uri(url);
            await webSocket.ConnectAsync(serverUri, token);
            while (webSocket.State == WebSocketState.Open && !token.IsCancellationRequested)
            {
                WebSocketReceiveResult result = await webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), token);
                if (result.MessageType == WebSocketMessageType.Close) break;
                string chunk = Encoding.UTF8.GetString(buffer, 0, result.Count); jsonAccumulator.Append(chunk);
                if (result.EndOfMessage)
                {
                    string rawStream = jsonAccumulator.ToString();
                    string[] packets = rawStream.Replace("}{", "}||{").Split(new string[] { "||" }, StringSplitOptions.RemoveEmptyEntries);
                    for (int i = 0; i < packets.Length; i++)
                    {
                        string singleJson = packets[i].Trim();
                        if (singleJson.StartsWith("{")) { lock (queueLock) { rawPacketQueue.Enqueue(singleJson); } }
                    }
                    jsonAccumulator.Clear();
                }
            }
        }
        catch { }
    }

    private byte[] buffer = new byte[32768]; 
    private StringBuilder jsonAccumulator = new StringBuilder();

    private void OnDestroy() { if (cts != null) { cts.Cancel(); cts.Dispose(); } if (webSocket != null) webSocket.Dispose(); }
}