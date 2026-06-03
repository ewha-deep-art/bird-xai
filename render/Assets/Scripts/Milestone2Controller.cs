using UnityEngine;
using UnityEngine.VFX;

public class Milestone2Controller : MonoBehaviour
{
    [Header("[1] REFERENCES (드래그 앤 드롭 결속 섹션)")]
    [Tooltip("현재 파티클이 이쁘게 뿜어져 나오고 있는 BirdFlockSystem 오브젝트를 넣으세요.")]
    public VisualEffect flockVFX; 
    [Tooltip("고생해서 제작하신 ArtInsideGridBox 오브젝트를 넣으세요.")]
    public Transform gridBoxTransform; 
    [Header("💡 추가된 항목")]
    [Tooltip("박스 안에서 함께 부드럽게 움직이게 만들 BirdLeader_Core 오브젝트를 넣으세요.")]
    public Transform leaderCoreTransform;

    [Header("[2] MOVEMENT SETTINGS (비행 속도 및 감속 제어)")]
    [Range(0.5f, 5.0f)]
    public float smoothTime = 2.0f; // 숫자가 클수록 묵직하고 부드럽게 추종
    public float maxSpeed = 20f;

    [Header("[3] LIVE MONITOR (실시간 데이터 계측기 섹션)")]
    [SerializeField] private string debugStatus = "게임 화면 클릭 후 [스페이스바]를 누르세요!";
    [SerializeField] private Vector3 liveTargetPos;  // 목적지 수치 모니터링
    [SerializeField] private Vector3 liveCurrentPos; // 부드럽게 변하는 현재 우두머리 수치 모니터링

    // 내부 연산 전용 가속도 및 방향 보관 변수
    private Vector3 velocity = Vector3.zero;
    private Vector3 lastPosition;

    void Start()
    {
        // 박스 중심 원점이나 현재 컨트롤러 위치에서 안전하게 출발
        Vector3 startPos = (gridBoxTransform != null) ? gridBoxTransform.position : Vector3.zero;
        liveTargetPos = startPos;
        liveCurrentPos = startPos;
        lastPosition = startPos;
        
        // 시작할 때 우두머리 코어 위치도 원점으로 강제 동기화
        if (leaderCoreTransform != null)
        {
            leaderCoreTransform.position = startPos;
        }
        
        Debug.Log("<color=yellow>[Milestone 2] 우두머리 코어 동기화 기능 로드 완료. 스페이스바를 입력하세요.</color>");
    }

    void Update()
    {
        // 1. 키보드 스페이스바를 누르면 박스 스케일(100, 50, 100) 내부의 임의의 위치로 타겟을 갱신 (더미 데이터 주입)
        if (Input.GetKeyDown(KeyCode.Space))
        {
            float randomX = Random.Range(-45f, 45f);
            float randomY = Random.Range(-20f, 20f);
            float randomZ = Random.Range(-45f, 45f);
            
            Vector3 center = (gridBoxTransform != null) ? gridBoxTransform.position : Vector3.zero;
            liveTargetPos = center + new Vector3(randomX, randomY, randomZ);
            
            debugStatus = "새로운 가상 GPS 수신 완료! 추적 시작.";
            Debug.Log($"<color=cyan>[DATA INPUT] 가상 데이터 수신 -> 목적지: {liveTargetPos}</color>");
        }

        // 2. [휙휙 튀기 방지] 현재 위치에서 타겟 좌표로 부드러운 유선형 감속 보간 이동 (SmoothDamp)
        liveCurrentPos = Vector3.SmoothDamp(liveCurrentPos, liveTargetPos, ref velocity, smoothTime, maxSpeed, Time.deltaTime);

        // 3. 💡 [추가] 우두머리 코어(BirdLeader_Core) 오브젝트를 부드러운 실시간 연산 좌표로 이동 및 비행 정렬
        if (leaderCoreTransform != null)
        {
            // 이동 처리
            leaderCoreTransform.position = liveCurrentPos;

            // 이동 방향 벡터 계산 및 회전각 정렬 (새가 날아가는 정면 바라보기)
            Vector3 moveDirection = liveCurrentPos - lastPosition;
            if (moveDirection.magnitude > 0.001f)
            {
                leaderCoreTransform.rotation = Quaternion.LookRotation(moveDirection.normalized);
            }
        }

        // 4. VFX 그래프의 Blackboard 변수 'LeaderPosition'으로 실시간 정규화 좌표 주입
        if (flockVFX != null)
        {
            flockVFX.SetVector3("LeaderPosition", liveCurrentPos);
        }

        // 다음 프레임 계산을 위해 현재 위치 백업
        lastPosition = liveCurrentPos;
    }
}