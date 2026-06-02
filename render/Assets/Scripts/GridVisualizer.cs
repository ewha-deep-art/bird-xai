using System.Collections.Generic;
using UnityEngine;

public class GridVisualizer : MonoBehaviour
{
    [Header("Environment Sprites (1~8)")]
    // 인스펙터창에서 1~8번 이미지(작물, 일조시간, 습도, 바람, 나무 등)를 순서대로 넣을 배열
    [SerializeField] private Sprite[] environmentSprites = new Sprite[8];

    [Header("Simulation Settings")]
    [SerializeField] private int viewRadius = 10; // 철새 주변 사방 몇 칸까지 격자를 그릴지 (10칸이면 가로세로 21x21)
    [SerializeField] private float gridSpacing = 1.0f; // 격자 간격 (카메라 Size 15.635, PPU 100 기준이므로 1.0이 딱 맞습니다)

    // 10초 데모용 철새의 가상 GPS 경로 데이터 (시간별 중심 좌표)
    private List<Vector2Int> mockBirdPath = new List<Vector2Int>();
    private float timer = 0f;
    private int currentFrameIndex = 0;
    private float timePerFrame = 1.0f; // 1초마다 철새가 다음 가상 좌표로 순간이동

    private void Start()
    {
        // 10초 분량의 데모용 가상 데이터 세팅 (X, Y 좌표계 예시)
        // 나중에 이 부분이 FastAPI에서 받아오는 실제 데이터 포인트로 대체됩니다.
        mockBirdPath.Add(new Vector2Int(100, 100)); // 0초
        mockBirdPath.Add(new Vector2Int(101, 102)); // 1초
        mockBirdPath.Add(new Vector2Int(103, 105)); // 2초
        mockBirdPath.Add(new Vector2Int(105, 108)); // 3초
        mockBirdPath.Add(new Vector2Int(108, 110)); // 4초
        mockBirdPath.Add(new Vector2Int(112, 111)); // 5초
        mockBirdPath.Add(new Vector2Int(115, 113)); // 6초
        mockBirdPath.Add(new Vector2Int(117, 116)); // 7초
        mockBirdPath.Add(new Vector2Int(120, 118)); // 8초
        mockBirdPath.Add(new Vector2Int(122, 122)); // 9초
        mockBirdPath.Add(new Vector2Int(125, 125)); // 10초

        UpdateGridVisualization();
    }

    private void Update()
    {
        timer += Time.deltaTime;

        // 1초가 지날 때마다 다음 시나리오 프레임으로 전환
        if (timer >= timePerFrame)
        {
            timer = 0f;
            currentFrameIndex++;

            if (currentFrameIndex < mockBirdPath.Count)
            {
                UpdateGridVisualization();
            }
            else
            {
                // 10초가 끝나면 무한 루프로 데모 재생을 위해 처음으로 리셋
                currentFrameIndex = 0;
                UpdateGridVisualization();
            }
        }
    }

    // 철새의 현재 위치를 기준점으로 삼아 주변 격자를 업데이트하는 핵심 함수
    private void UpdateGridVisualization()
    {
        // 1. 기존에 켜져 있던 모든 격자들을 오브젝트 풀에 다시 반납(숨김)
        ObjectPooler.Instance.ResetAllCells();

        Vector2Int birdPos = mockBirdPath[currentFrameIndex];

        // 2. 철새 주변 (birdPos.x - viewRadius) 부터 (birdPos.x + viewRadius) 까지 바둑판 루프 순회
        for (int x = -viewRadius; x <= viewRadius; x++)
        {
            for (int y = -viewRadius; y <= viewRadius; y++)
            {
                // 현재 그리고 있는 가상 격자의 절대 좌표
                int currentX = birdPos.x + x;
                int currentY = birdPos.y + y;

                // --- [미디어아트 연출 로직 스폿] ---
                // 지금은 외부 데이터가 없으므로 가상의 규칙(수식 구동)으로 픽셀을 생성합니다.
                // 이 수식 규칙은 나중에 FastAPI 데이터 가공 로직으로 완벽하게 대체 가능합니다.
                
                // 가상 풍속 단계 계산 (10, 30, 50, 70, 100 단계 연출용 수식)
                int pseudoWind = Mathf.Abs((currentX + currentY) % 5); // 0, 1, 2, 3, 4 중 하나 반환
                float[] windScales = { 0.15f, 0.35f, 0.55f, 0.75f, 1.0f }; // 조건 설명에 기반한 5단계 크기 비율
                float targetScale = windScales[pseudoWind];

                // 가상 환경 요소 타입 결정 (1~8번 이미지 매핑용 인스펙터 인덱스)
                int pseudoEnvIndex = Mathf.Abs((currentX * currentY) % 8); // 0~7 중 하나 반환

                // 3. 오브젝트 풀에서 대기 중인 GridCell을 하나 꺼내옵니다.
                GameObject cell = ObjectPooler.Instance.GetPooledObject();
                if (cell == null) return;

                // 4. 철새 중심 기준 상대적 위치 계산하여 배치 (유니티 화면 정중앙 (0,0) 주변에 이쁘게 모이도록 정렬)
                cell.transform.position = new Vector3(x * gridSpacing, y * gridSpacing, 0);

                // 5. 1~8번 이미지 중 데이터 결과에 맞는 스프라이트 할당
                SpriteRenderer renderer = cell.GetComponent<SpriteRenderer>();
                if (renderer != null && pseudoEnvIndex < environmentSprites.Length)
                {
                    renderer.sprite = environmentSprites[pseudoEnvIndex];
                }

                // 6. 풍속 데이터에 맞게 크기(Scale) 조정
                cell.transform.localScale = new Vector3(targetScale, targetScale, 1f);

                // 7. 준비가 완료된 격자 오브젝트를 화면에 활성화
                cell.SetActive(true);
            }
        }
    }
}