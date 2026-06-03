using UnityEngine;
using UnityEngine.VFX;

public class BirdNetworkManager : MonoBehaviour
{
    [Header("References")]
    public VisualEffect flockVFX;      // 유니티 VFX 컴포넌트
    public Transform leaderVisual;    // 우두머리 위치 시각화용 (Sphere 등)
    public Transform gridBoxTransform; // 고생해서 만드신 ArtInsideGridBox를 여기에 드래그 앤 드롭하세요.

    [Header("Movement Smooth Settings")]
    [Tooltip("숫자가 작을수록 목표 지점까지 부드럽고 묵직하게 이동합니다. (추천: 1.5 ~ 3.0)")]
    public float smoothTime = 2.0f;
    [Tooltip("철새 우두머리의 최대 이동 속도 제한")]
    public float maxSpeed = 15f;

    // 서버 데이터 경계값 (제시해주신 데이터 기준)
    private const float MinLat = -5.7379746f;
    private const float MaxLat = 43.662685f;
    private const float MinLon = -76.26561f;
    private const float MaxLon = -48.195133f;
    private const float MinAlt = -47.35702f;
    private const float MaxAlt = 235.4282f;

    // 내부 연산 변수
    private Vector3 targetNormalizedPosition; // 서버 데이터가 도달해야 할 최종 목적지
    private Vector3 currentVelocity = Vector3.zero;
    private Vector3 currentPosition;          // 현재 철새 우두머리의 부드러운 실시간 위치

    [Header("Debug Monitor (실시간 데이터 확인 창)")]
    [SerializeField] private string rawServerDataLog = "No Data Yet";
    [SerializeField] private Vector3 liveTargetPos;
    [SerializeField] private Vector3 liveCurrentPos;

    void Start()
    {
        // 시작할 때는 원점(혹은 박스 중심)에서 안전하게 출발합니다.
        if (gridBoxTransform != null)
        {
            currentPosition = gridBoxTransform.position;
            targetNormalizedPosition = gridBoxTransform.position;
        }
        else
        {
            currentPosition = Vector3.zero;
            targetNormalizedPosition = Vector3.zero;
        }
    }

    /// <summary>
    /// FastAPI 서버로부터 1초에 한 번씩 호출되는 데이터 수신 함수
    /// </summary>
    public void OnReceiveServerData(float lat, float lon, float altitude, float tailwindAttr, float headwindAttr)
    {
        // 1. 디버그용 원본 로그 기록
        rawServerDataLog = $"[RAW] Lat: {lat} | Lon: {lon} | Alt: {altitude}";

        // 2. 서버 데이터를 0 ~ 1 범위로 정규화 (InverseLerp)
        float normX = Mathf.InverseLerp(MinLon, MaxLon, lon);  // 경도 -> X
        float normY = Mathf.InverseLerp(MinAlt, MaxAlt, altitude); // 고도 -> Y
        float normZ = Mathf.InverseLerp(MinLat, MaxLat, lat);  // 위도 -> Z

        // 3. ArtInsideGridBox 스케일(100, 50, 100) 내부 영역으로 맵핑
        // 박스 벽면에 완전히 부딪히는 것을 막기 위해 상하좌우 약 10%의 패딩(여백)을 둡니다.
        float boxWidth = 100f;
        float boxHeight = 50f;
        float boxLength = 100f;

        float worldX = Mathf.Lerp(-boxWidth * 0.4f, boxWidth * 0.4f, normX);
        float worldY = Mathf.Lerp(-boxHeight * 0.4f, boxHeight * 0.4f, normY);
        float worldZ = Mathf.Lerp(-boxLength * 0.4f, boxLength * 0.4f, normZ);

        // 4. 새로운 목표 좌표 설정 (ArtInsideGridBox의 중심 위치 기준 상대 좌표)
        Vector3 boxCenter = (gridBoxTransform != null) ? gridBoxTransform.position : Vector3.zero;
        targetNormalizedPosition = boxCenter + new Vector3(worldX, worldY, worldZ);
        
        liveTargetPos = targetNormalizedPosition; // 인스펙터 모니터링용

        // 5. XAI 바람 기여도 데이터 바로 주입
        if (flockVFX != null)
        {
            flockVFX.SetFloat("TailwindIntensity", tailwindAttr);
            flockVFX.SetFloat("HeadwindIntensity", headwindAttr);
        }
    }

    void Update()
    {
        // 6. [핵심] 매 프레임마다 현재 위치에서 목표 위치로 휙휙 바뀌지 않고, 스무스하게 보간이동시킵니다.
        // SmoothDamp는 시간에 구애받지 않고 가속/감속을 적용해 유기적인 생명체의 움직임을 만듭니다.
        currentPosition = Vector3.SmoothDamp(currentPosition, targetNormalizedPosition, ref currentVelocity, smoothTime, maxSpeed, Time.deltaTime);
        
        liveCurrentPos = currentPosition; // 인스펙터 모니터링용

        // 7. 실시간 좌표 들어오는지 콘솔 및 디버그 창에 띄우기 (11번 요구사항)
        // 콘솔 창이 너무 도배되는 것을 막으면서도 실시간 값이 바뀌는 것을 눈으로 볼 수 있습니다.
        if (Time.frameCount % 30 == 0) // 약 0.5초마다 한 번씩 콘솔 출력
        {
            Debug.Log($"[실시간 디버그] 목표 좌표: {targetNormalizedPosition} ==> 현재 부드러운 위치: {currentPosition}");
        }

        // 8. 실제 오브젝트 및 VFX에 부드러운 좌표 주입
        if (leaderVisual != null)
        {
            leaderVisual.position = currentPosition;
        }

        if (flockVFX != null)
        {
            flockVFX.SetVector3("LeaderPosition", currentPosition);
        }

        // --- 임시 테스트용 키 플러그인 (서버 통신 전 테스트용) ---
        if (Input.GetKeyDown(KeyCode.Space))
        {
            // 스페이스바를 누르면 박스 내부 임의의 장소로 타겟을 변경하여 부드럽게 흘러가는지 테스트 가능
            float fakeLat = Random.Range(MinLat, MaxLat);
            float fakeLon = Random.Range(MinLon, MaxLon);
            float fakeAlt = Random.Range(MinAlt, MaxAlt);
            OnReceiveServerData(fakeLat, fakeLon, fakeAlt, 0.7f, 0.1f);
            Debug.Log("<color=yellow>테스트용 가상 데이터가 주입되었습니다.</color>");
        }
    }
}