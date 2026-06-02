using System.Collections.Generic;
using UnityEngine;

public class ObjectPooler : MonoBehaviour
{
    // 싱글톤 패턴: 어디서나 이 오브젝트 풀에 쉽게 접근할 수 있도록 설정
    public static ObjectPooler Instance;

    [Header("Pool Settings")]
    [SerializeField] private GameObject gridCellPrefab; // 지난 단계에서 만든 GridCell 프리팹
    [SerializeField] private int poolSize = 900;         // 미리 생성해둘 개수 (30x30)

    // 생성된 격자 오브젝트들을 담아둘 리스트
    private List<GameObject> pooledObjects = new List<GameObject>();

    private void Awake()
    {
        Instance = this;
    }

    private void Start()
    {
        // 게임이 시작되자마자 정해진 개수만큼 격자를 미리 생성해서 숨김(비활성화)
        for (int i = 0; i < poolSize; i++)
        {
            GameObject obj = Instantiate(gridCellPrefab);
            obj.transform.SetParent(this.transform); // 계층 구조 정리를 위해 이 오브젝트의 자식으로 설정
            obj.SetActive(false);                   // 일단 화면에서 숨김
            pooledObjects.Add(obj);                 // 리스트에 보관
        }

        Debug.Log($"{poolSize}개의 GridCell 프리팹이 풀에 성공적으로 생성되었습니다.");
    }

    // 외부(철새 배치 스크립트 등)에서 필요할 때 격자 오브젝트를 하나씩 꺼내가는 함수
    public GameObject GetPooledObject()
    {
        // 리스트를 돌면서 현재 사용 중이지 않은(비활성화된) 오브젝트를 찾아서 반환
        for (int i = 0; i < pooledObjects.Count; i++)
        {
            if (!pooledObjects[i].activeInHierarchy)
            {
                return pooledObjects[i];
            }
        }

        // 만약 900개를 다 썼는데 더 필요하다면 비상용으로 새로 하나 만들어서 반환
        GameObject obj = Instantiate(gridCellPrefab);
        obj.transform.SetParent(this.transform);
        obj.SetActive(false);
        pooledObjects.Add(obj);
        return obj;
    }

    // 화면에 켜져 있는 모든 격자들을 한 번에 다시 숨기는(초기화) 함수
    public void ResetAllCells()
    {
        for (int i = 0; i < pooledObjects.Count; i++)
        {
            pooledObjects[i].SetActive(false);
        }
    }
}