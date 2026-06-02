using System.Collections.Generic;
using UnityEngine;
using NativeWebSocket;

// ── 서버 → Unity 최종 데이터 계약 명세 ────────────────────────
[System.Serializable]
public class LatLonPoint
{
    public float lat;
    public float lon;
    public float altitude_m;
}

[System.Serializable]
public class Candidate
{
    public string path_id;
    public float score;
}

[System.Serializable]
public class XaiAttributions
{
    public float daylength_h; 
    public float ws_925;      
    public float q_850;       
}

[System.Serializable]
public class XaiData
{
    public XaiAttributions attributions; 
}

[System.Serializable]
public class BoidAgent
{
    public string agent_id;
    public LatLonPoint position;
}

[System.Serializable]
public class FrameMessage
{
    public string schema_version;
    public string message_type;
    public LatLonPoint position;
    public LatLonPoint[] predicted_path;
    public Candidate[] candidates;
    public XaiData xai;
    public XaiAttributions attributions; 
    public BoidAgent[] boids;
    public string applied_overrides;
}

// ── WebSocket 초경량 지우개 오버라이드 렌더링 클래스 ──────────────────
public class Unity_WebSocket : MonoBehaviour
{
    private WebSocket ws;

    [Header("Network")]
    public string url = "ws://127.0.0.1:8000/ws";

    [Header("Environment Sprites (0~7)")]
    [Tooltip("Element 0: 검은색 지우개, Element 3: 광주기, Element 4: 대장, Element 5: 부하새, Element 6: 순풍, Element 7: 습도")]
    [SerializeField] private Sprite[] environmentSprites = new Sprite[8]; 

    [Header("Coordinate Mapping Settings")]
    [SerializeField] private float coordinateScale = 1.5f; 

    private Vector3 originGpsPosition; 
    private bool isOriginSet = false;

    [Header("Art Display Settings")]
    [SerializeField] private float xOffset = 0f;     
    [SerializeField] private float yOffset = -5f;    
    [SerializeField] private int maxHistoryCount = 6; 

    // ─── [정밀 자취 메모리 독립 뱅크 인덱싱] ──────────────────────────
    private List<Vector3> leaderHistoryPath = new List<Vector3>(); 
    private List<Vector3>[] followerHistoryPaths = new List<Vector3>[15]; 
    
    private List<List<Vector3>> windHistoryFrames = new List<List<Vector3>>();
    private List<List<Vector3>> humidityHistoryFrames = new List<List<Vector3>>();
    private List<List<Vector3>> lightHistoryFrames = new List<List<Vector3>>();

    async void Start()
    {
        for (int i = 0; i < 15; i++) followerHistoryPaths[i] = new List<Vector3>();

        ws = new WebSocket(url);

        ws.OnOpen  += () => Debug.Log("Bird-XAI 서버 연결 성공 - 초경량 지우개 오버라이드 엔진 가동");
        ws.OnClose += (e) => Debug.Log("Bird-XAI 서버 연결 끊김");
        ws.OnError += (e) => Debug.LogError("웹소켓 통신 에러 단속: " + e);

        ws.OnMessage += (bytes) =>
        {
            string json = System.Text.Encoding.UTF8.GetString(bytes);
            FrameMessage msg = JsonUtility.FromJson<FrameMessage>(json);

            if (msg != null && msg.message_type == "frame")
            {
                RenderBirdFrame(msg);
            }
        };

        await ws.Connect();
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        ws.DispatchMessageQueue();
#endif
    }

    private void RenderBirdFrame(FrameMessage msg)
    {
        // [최적화 극대화]: 수천 칸의 타일맵을 지우고 갱신하던 무거운 연산 완전 삭제!
        // 오직 화면에 찍히는 수십 개의 실체 오브젝트 풀 세포들만 리셋합니다.
        ObjectPooler.Instance.ResetAllCells();

        if (msg.position == null) return;

        if (!isOriginSet)
        {
            originGpsPosition = new Vector3(msg.position.lon, msg.position.lat, 0);
            isOriginSet = true;
            Debug.Log("[시스템] 미디어월 기준 원점 고정 완료");
        }

        // ──────────────────────────────────────────────────────────────
        // 2. [우두머리 철새 (Leader)] 격자 연산 및 자취 출력
        // ──────────────────────────────────────────────────────────────
        float leaderRawX = (msg.position.lon - originGpsPosition.x) * coordinateScale;
        float leaderRawY = (msg.position.lat - originGpsPosition.y) * coordinateScale;

        int leaderGridX = Mathf.RoundToInt((leaderRawX + xOffset) / 0.3f);
        int leaderGridY = Mathf.RoundToInt((leaderRawY + yOffset) / 0.3f);
        Vector3 currentLeaderGridPos = new Vector3(leaderGridX * 0.3f, leaderGridY * 0.3f, 0);

        currentLeaderGridPos.x = Mathf.Clamp(currentLeaderGridPos.x, -8.0f, 8.0f);
        currentLeaderGridPos.y = Mathf.Clamp(currentLeaderGridPos.y, -15.0f, 18.0f);

        if (leaderHistoryPath.Count == 0 || leaderHistoryPath[leaderHistoryPath.Count - 1] != currentLeaderGridPos)
        {
            leaderHistoryPath.Add(currentLeaderGridPos);
            if (leaderHistoryPath.Count > maxHistoryCount) leaderHistoryPath.RemoveAt(0); 
        }

        Vector2 leaderTargetPos = new Vector2(currentLeaderGridPos.x, currentLeaderGridPos.y);

        // ◀ [역발상 오버라이드] 대장 새 본체가 찍히기 바로 밑단에 검은색 지우개 픽셀을 먼저 깔아 배경을 가립니다.
        DrawEraserCell(currentLeaderGridPos);

        GameObject leaderCell = ObjectPooler.Instance.GetPooledObject();
        if (leaderCell != null)
        {
            leaderCell.transform.position = currentLeaderGridPos;
            SpriteRenderer renderer = leaderCell.GetComponent<SpriteRenderer>();
            if (renderer != null && environmentSprites.Length > 4)
            {
                renderer.sprite = environmentSprites[4]; 
                renderer.sortingOrder = 15; 
                renderer.color = Color.white;
            }
            leaderCell.SetActive(true);
        }

        RenderPixelTrail(leaderHistoryPath, 4, 14);

        // ──────────────────────────────────────────────────────────────
        // 3. [추종자 무리] 격자 연산 및 자취 출력
        // ──────────────────────────────────────────────────────────────
        Vector2 leaderDirection = Vector2.up; 
        if (msg.predicted_path != null && msg.predicted_path.Length > 0)
        {
            float pX = (msg.predicted_path[0].lon - msg.position.lon) * coordinateScale;
            float pY = (msg.predicted_path[0].lat - msg.position.lat) * coordinateScale;
            if (new Vector2(pX, pY).sqrMagnitude > 0.001f) leaderDirection = new Vector2(pX, pY).normalized;
        }

        Vector2 leftWingDir = Quaternion.Euler(0, 0, 135) * leaderDirection;  
        Vector2 rightWingDir = Quaternion.Euler(0, 0, -135) * leaderDirection; 

        for (int i = 0; i < 15; i++)
        {
            Vector2 assignedWing = (i % 2 == 0) ? leftWingDir : rightWingDir;
            float slotDepth = (i / 2) * 0.35f + 0.25f; 
            Vector2 instantFollowerPos = leaderTargetPos + assignedWing * slotDepth;

            instantFollowerPos.x = Mathf.Clamp(instantFollowerPos.x, -8.3f, 8.3f);
            instantFollowerPos.y = Mathf.Clamp(instantFollowerPos.y, -15.3f, 18.3f);

            int fGridX = Mathf.RoundToInt(instantFollowerPos.x / 0.3f);
            int fGridY = Mathf.RoundToInt(instantFollowerPos.y / 0.3f);
            Vector3 currentFollowerGridPos = new Vector3(fGridX * 0.3f, fGridY * 0.3f, 0);

            List<Vector3> fHistory = followerHistoryPaths[i];
            if (fHistory.Count == 0 || fHistory[fHistory.Count - 1] != currentFollowerGridPos)
            {
                fHistory.Add(currentFollowerGridPos);
                if (fHistory.Count > maxHistoryCount) fHistory.RemoveAt(0);
            }

            // ◀ [역발상 오버라이드] 부하 새 본체가 찍히기 밑단에도 검은색 지우개를 깔아 배경을 가립니다.
            DrawEraserCell(currentFollowerGridPos);

            GameObject followerCell = ObjectPooler.Instance.GetPooledObject();
            if (followerCell != null)
            {
                followerCell.transform.position = currentFollowerGridPos;
                SpriteRenderer renderer = followerCell.GetComponent<SpriteRenderer>();
                if (renderer != null && environmentSprites.Length > 5)
                {
                    renderer.sprite = environmentSprites[5]; 
                    renderer.sortingOrder = 20; 
                    renderer.color = Color.white;
                }
                followerCell.SetActive(true);
            }

            RenderPixelTrail(fHistory, 5, 13);
        }

        // ──────────────────────────────────────────────────────────────
        // 4. [XAI 핵심 연출] - 안개형 분사 및 배경 오버라이드
        // ──────────────────────────────────────────────────────────────
        XaiAttributions attrs = null;
        if (msg.xai != null && msg.xai.attributions != null) attrs = msg.xai.attributions;
        else if (msg.attributions != null) attrs = msg.attributions;

        if (attrs != null)
        {
            int originX = leaderGridX;
            int originY = leaderGridY;

            List<Vector2Int> localGridArea = new List<Vector2Int>();
            int maxRadiusCells = 22;

            for (int dx = -maxRadiusCells; dx <= maxRadiusCells; dx++)
            {
                for (int dy = -maxRadiusCells; dy <= maxRadiusCells; dy++)
                {
                    float distance = Mathf.Sqrt(dx * dx + dy * dy);
                    float spawnProbability = Mathf.Clamp01(1.0f - (distance / maxRadiusCells));

                    if (UnityEngine.Random.value < spawnProbability)
                    {
                        localGridArea.Add(new Vector2Int(originX + dx, originY + dy));
                    }
                }
            }
            int totalCells = localGridArea.Count;

            for (int i = 0; i < localGridArea.Count; i++)
            {
                Vector2Int temp = localGridArea[i];
                int randomIndex = UnityEngine.Random.Range(i, localGridArea.Count);
                localGridArea[i] = localGridArea[randomIndex];
                localGridArea[randomIndex] = temp;
            }

            float balancedLightWeight = attrs.daylength_h * 0.03f; 
            float balancedHumidityWeight = attrs.q_850 * 0.2f;     

            int windCount     = Mathf.RoundToInt(Mathf.Clamp01(attrs.ws_925) * totalCells);
            int humidityCount = Mathf.RoundToInt(Mathf.Clamp01(balancedHumidityWeight) * totalCells);
            int lightCount    = Mathf.RoundToInt(Mathf.Clamp01(balancedLightWeight) * totalCells);

            int currentPointer = 0;
            float baseEnvAlpha = 0.8f;

            // ① 순풍(ws_925) 배치 및 지우개 가동
            List<Vector3> currentFrameWinds = new List<Vector3>();
            for (int i = 0; i < windCount && currentPointer < localGridArea.Count; i++)
            {
                int gx = localGridArea[currentPointer].x; int gy = localGridArea[currentPointer].y;
                Vector3 wPos = new Vector3(gx * 0.3f, gy * 0.3f, 0);
                wPos.x = Mathf.Clamp(wPos.x, -8.3f, 8.3f); wPos.y = Mathf.Clamp(wPos.y, -15.3f, 18.3f);
                currentFrameWinds.Add(wPos);

                // 환경 알갱이 본체가 찍히는 바닥 격자에 검은색 지우개를 먼저 배치
                DrawEraserCell(wPos);
                DrawEnvironmentCell(wPos, 6, 5, baseEnvAlpha); 
                currentPointer++;
            }

            // ② 습도(q_850) 배치 및 지우개 가동
            List<Vector3> currentFrameHumidities = new List<Vector3>();
            for (int i = 0; i < humidityCount && currentPointer < localGridArea.Count; i++)
            {
                int gx = localGridArea[currentPointer].x; int gy = localGridArea[currentPointer].y;
                Vector3 hPos = new Vector3(gx * 0.3f, gy * 0.3f, 0);
                hPos.x = Mathf.Clamp(hPos.x, -8.3f, 8.3f); hPos.y = Mathf.Clamp(hPos.y, -15.3f, 18.3f);
                currentFrameHumidities.Add(hPos);

                DrawEraserCell(hPos);
                DrawEnvironmentCell(hPos, 7, 4, baseEnvAlpha); 
                currentPointer++;
            }

            // ③ 광주기(daylength_h) 배치 및 지우개 가동
            List<Vector3> currentFrameLights = new List<Vector3>();
            for (int i = 0; i < lightCount && currentPointer < localGridArea.Count; i++)
            {
                int gx = localGridArea[currentPointer].x; int gy = localGridArea[currentPointer].y;
                Vector3 lPos = new Vector3(gx * 0.3f, gy * 0.3f, 0);
                lPos.x = Mathf.Clamp(lPos.x, -8.3f, 8.3f); lPos.y = Mathf.Clamp(lPos.y, -15.3f, 18.3f);
                currentFrameLights.Add(lPos);

                DrawEraserCell(lPos);
                DrawEnvironmentCell(lPos, 3, 3, baseEnvAlpha); 
                currentPointer++;
            }
            
            RenderEnvironmentPixelTrail(currentFrameWinds, windHistoryFrames, 6, 5);
            RenderEnvironmentPixelTrail(currentFrameHumidities, humidityHistoryFrames, 7, 4);
            RenderEnvironmentPixelTrail(currentFrameLights, lightHistoryFrames, 3, 3);
        }
    }

    // ─── [초경량 최적화의 핵심 핵심: 지우개 스프라이트 드로어] ──────────────────────────
    private void DrawEraserCell(Vector3 position)
    {
        GameObject eraserCell = ObjectPooler.Instance.GetPooledObject();
        if (eraserCell == null) return;

        eraserCell.transform.position = position;
        SpriteRenderer renderer = eraserCell.GetComponent<SpriteRenderer>();
        if (renderer != null && environmentSprites.Length > 0)
        {
            renderer.sprite = environmentSprites[0]; // Element 0 (검은색 사각형 지우개)
            renderer.sortingOrder = 2; // 통배경(1)보다는 위, 철새/환경알갱이(3~20)보다는 아래에 위치하여 풀을 완벽히 덮어 가림
            renderer.color = Color.white;
        }
        eraserCell.SetActive(true);
    }

    private void DrawEnvironmentCell(Vector3 position, int elementIndex, int sortingOrder, float alpha)
    {
        GameObject cell = ObjectPooler.Instance.GetPooledObject();
        if (cell == null) return;
        cell.transform.position = position;
        SpriteRenderer renderer = cell.GetComponent<SpriteRenderer>();
        if (renderer != null && environmentSprites.Length > elementIndex)
        {
            renderer.sprite = environmentSprites[elementIndex];
            renderer.sortingOrder = sortingOrder;
            renderer.color = new Color(1f, 1f, 1f, alpha);
        }
        cell.SetActive(true);
    }

    private void RenderEnvironmentPixelTrail(List<Vector3> currentFramePositions, List<List<Vector3>> historyFrames, int elementIndex, int baseSortingOrder)
    {
        historyFrames.Add(new List<Vector3>(currentFramePositions));
        if (historyFrames.Count > maxHistoryCount) historyFrames.RemoveAt(0);

        int totalStoredFrames = historyFrames.Count;
        if (totalStoredFrames <= 1) return;

        for (int i = totalStoredFrames - 2; i >= 0; i--)
        {
            int stepsBack = (totalStoredFrames - 1) - i;
            float calculatedAlpha = 0.8f * Mathf.Pow(0.5f, stepsBack); 

            if (calculatedAlpha < 0.01f) continue;

            List<Vector3> pastFramePoints = historyFrames[i];
            foreach (Vector3 pastPos in pastFramePoints)
            {
                // 환경 알갱이들의 과거 잔상이 남는 자리도 지우개 세포를 깔아서 풀밭을 실시간 오버라이드
                DrawEraserCell(pastPos);
                DrawEnvironmentCell(pastPos, elementIndex, baseSortingOrder - 1, calculatedAlpha);
            }
        }
    }

    private void RenderPixelTrail(List<Vector3> pathHistory, int spriteElementIndex, int baseSortingOrder)
    {
        if (pathHistory.Count <= 1) return;
        int historyLength = pathHistory.Count;

        for (int i = historyLength - 2; i >= 0; i--)
        {
            GameObject trailCell = ObjectPooler.Instance.GetPooledObject();
            if (trailCell == null) continue;

            trailCell.transform.position = pathHistory[i];
            
            // 철새 무리들의 과거 자취 꼬리가 남는 자리도 지우개 세포를 밑단에 깔아 배경 삭제
            DrawEraserCell(pathHistory[i]);

            SpriteRenderer renderer = trailCell.GetComponent<SpriteRenderer>();
            if (renderer != null && environmentSprites.Length > spriteElementIndex)
            {
                renderer.sprite = environmentSprites[spriteElementIndex];
                renderer.sortingOrder = baseSortingOrder;

                int stepsBack = (historyLength - s1) - i; 
                float calculatedAlpha = Mathf.Pow(0.5f, stepsBack); 

                if (calculatedAlpha < 0.01f) continue; 
                renderer.color = new Color(1f, 1f, 1f, calculatedAlpha);
            }
            trailCell.SetActive(true);
        }
    }

    async void OnDestroy()
    {
        if (ws != null) await ws.Close();
    }
}