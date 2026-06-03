using System.Collections.Generic;
using UnityEngine;
using UnityEngine.VFX; // 💡 VFX Graph 내부 버퍼 직결을 위해 필수 추가

public class BoidsController : MonoBehaviour
{
    [Header("[1] BOIDS DESIGN SETTINGS")]
    public int boidsCount = 1000; // 💡 제공된 이미지 기반 기본 마리수 1000마리 락온
    public float minSpeed = 15f;  // 💡 기류 결 형성을 위해 하한 속도 상향
    public float maxSpeed = 35f;  // 💡 벽면 돌파를 막기 위한 상한선 정렬

    [Header("[2] BOIDS WEIGHTS (기획안 저격 군집 계수)")]
    public float cohesionWeight = 8.5f;   // 💡 한 가닥의 선형 띠로 묶기 위해 응집력 폭발적 상향
    public float separationWeight = 0.2f; // 💡 사방으로 모기떼처럼 찢어지는 현상 차단
    public float alignmentWeight = 4.5f;  // 💡 칼같은 결(Streamline)을 유지하기 위해 정렬력 상향
    
    [Header("[3] RADIUS SETTINGS")]
    public float neighborRadius = 45f;    // 💡 광활한 은하수 편대를 위해 인지 반경 확장
    public float separationRadius = 1.2f; // 💡 압축된 선형 밀도를 위해 분리 반경 축소

    [HideInInspector] public Vector3[] boidsPositions;
    private Vector3[] boidsVelocities;

    // ── 💡 변수명 결속 팩트 체크 완료 ──
    private BirdDataManager birdDataManager;
    private VisualEffect cachedVFX;

    void Start()
    {
        InitBoids();
    }

    void InitBoids()
    {
        boidsPositions = new Vector3[boidsCount];
        boidsVelocities = new Vector3[boidsCount];

        for (int i = 0; i < boidsCount; i++)
        {
            // ── 💡 기획안 매칭: 선형(Line) 스폰 공식 ──
            // 구체 모양으로 소환되지 않고 가로로 늘어선 얇은 기류 띠 형태로 철새들이 태어납니다.
            float spawnX = Random.Range(-35f, 35f); 
            float spawnY = Random.Range(-5f, 5f);   
            float spawnZ = Random.Range(-5f, 5f);   
            
            boidsPositions[i] = new Vector3(spawnX, spawnY, spawnZ);

            // ── 💡 기획안 매칭: 초기 방향성 기류 주입 ──
            // 시작하자마자 사방으로 흩어지지 않고 우측 사선 궤적으로 질주를 시작합니다.
            Vector3 streamDirection = new Vector3(1.0f, 0.2f, 0.5f).normalized;
            float randomSpeed = Random.Range(minSpeed, maxSpeed * 0.8f);
            boidsVelocities[i] = streamDirection * randomSpeed;
        }
    }

    void Update()
    {
        // 실시간 컴포넌트 자동 탐색 및 캐싱 가동
        if (birdDataManager == null) birdDataManager = FindObjectOfType<BirdDataManager>();
        if (cachedVFX == null) cachedVFX = FindObjectOfType<VisualEffect>();

        // 성능 최적화: 프레임 분산 연산 (에디터 렉 방지)
        if (Time.frameCount % 2 != 0) return;

        if (boidsPositions == null || boidsPositions.Length != boidsCount)
        {
            InitBoids();
        }

        Vector3 targetLeaderPos = Vector3.zero;
        Vector3 leaderForwardDir = Vector3.forward;

        if (birdDataManager != null && birdDataManager.leaderCoreTransform != null)
        {
            // 우두머리 새의 정면 시선 벡터 동기화
            leaderForwardDir = birdDataManager.leaderCoreTransform.forward;
            
            // 후방 2m 편대 비행용 가상 추격점 설정
            if (leaderForwardDir.magnitude < 0.001f)
            {
                targetLeaderPos = birdDataManager.leaderCoreTransform.position - new Vector3(0f, 0f, 2.0f);
            }
            else
            {
                targetLeaderPos = birdDataManager.leaderCoreTransform.position - (leaderForwardDir.normalized * 2.0f);
            }
        }

        // 보이드 물리 연산 코어 루프
        for (int i = 0; i < boidsCount; i++)
        {
            Vector3 cohesion = Vector3.zero;
            Vector3 separation = Vector3.zero;
            Vector3 alignment = Vector3.zero;
            int neighborsCount = 0;

            for (int j = 0; j < boidsCount; j++)
            {
                if (i == j) continue;

                float distance = Vector3.Distance(boidsPositions[i], boidsPositions[j]);

                if (distance < neighborRadius)
                {
                    cohesion += boidsPositions[j];
                    alignment += boidsVelocities[j];
                    neighborsCount++;
                }

                if (distance < separationRadius)
                {
                    separation += (boidsPositions[i] - boidsPositions[j]) / distance;
                }
            }

            Vector3 leaderFollowForce = (targetLeaderPos - boidsPositions[i]).normalized;

            if (neighborsCount > 0)
            {
                cohesion = (cohesion / neighborsCount) - boidsPositions[i];
                alignment = alignment / neighborsCount;
            }

            // ── 💡 선형 결 저격 물리 가속도 결합 수식 ──
            // 기본 보이드 힘에 우두머리의 시선(Forward) 정렬 힘을 강력하게 병합합니다.
            Vector3 acceleration = (cohesion * cohesionWeight) + 
                                  (separation * separationWeight) + 
                                  (alignment * alignmentWeight) + 
                                  (leaderFollowForce * 10.0f) +
                                  (leaderForwardDir * 18.0f); // 새떼들이 우두머리 고개 방향으로 칼같이 궤적을 그리게 유도

            // 유선형 경계면 반사 소프트 쿠션 물리 (상자 벽면 박힘 현상 원천 차단)
            float bBoxW = (100f * 0.5f) * 0.85f; 
            float bBoxH = (50f * 0.5f) * 0.85f;  
            float bBoxL = (100f * 0.5f) * 0.85f; 

            float turnBoundaryMargin = 6.0f; // 벽면 도달 6m 전방부터 유턴 작동
            Vector3 avoidForce = Vector3.zero;

            if (boidsPositions[i].x > bBoxW - turnBoundaryMargin) avoidForce.x = -1f;
            else if (boidsPositions[i].x < -bBoxW + turnBoundaryMargin) avoidForce.x = 1f;

            if (boidsPositions[i].y > bBoxH - turnBoundaryMargin) avoidForce.y = -1f;
            else if (boidsPositions[i].y < -bBoxH + turnBoundaryMargin) avoidForce.y = 1f;

            if (boidsPositions[i].z > bBoxL - turnBoundaryMargin) avoidForce.z = -1f;
            else if (boidsPositions[i].z < -bBoxL + turnBoundaryMargin) avoidForce.z = 1f;

            if (avoidForce.magnitude > 0.1f)
            {
                acceleration += avoidForce.normalized * 40.0f; // 중앙 강제 튕겨냄 물리 작동
            }

            // 최종 물리 프레임 가산 연산
            boidsVelocities[i] += acceleration * Time.deltaTime;
            boidsVelocities[i] = Vector3.ClampMagnitude(boidsVelocities[i], maxSpeed);
            boidsPositions[i] += boidsVelocities[i] * Time.deltaTime;

            // 하드 이탈 가드 락
            boidsPositions[i].x = Mathf.Clamp(boidsPositions[i].x, -bBoxW, bBoxW);
            boidsPositions[i].y = Mathf.Clamp(boidsPositions[i].y, -bBoxH, bBoxH);
            boidsPositions[i].z = Mathf.Clamp(boidsPositions[i].z, -bBoxL, bBoxL);
        }

        // ── 💡 GPU VFX GRAPH 직결 패스 엔진 (여기에 줄끈이 닿아야 파티클이 소환됩니다) ──
        if (cachedVFX != null && boidsPositions != null)
        {
            Texture2D boidsTex = new Texture2D(boidsCount, 1, TextureFormat.RGBAFloat, false);
            for (int i = 0; i < boidsCount; i++)
            {
                Vector3 pos = boidsPositions[i];
                Color posColor = new Color(pos.x, pos.y, pos.z, 1.0f);
                boidsTex.SetPixel(i, 0, posColor);
            }
            boidsTex.Apply();

            cachedVFX.SetTexture("BoidsPositionTexture", boidsTex); // 인스펙터 노출 키 동기화
            cachedVFX.SetInt("BoidsCount", boidsCount);             // 인스펙터 노출 키 동기화
        }
    }
}