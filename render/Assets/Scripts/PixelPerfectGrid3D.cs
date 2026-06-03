using UnityEngine;

public class PixelPerfectGrid3D : MonoBehaviour
{
    [Header("가상 무대 규격 설정 (100, 50, 100)")]
    public Vector3 stageSize = new Vector3(100f, 50f, 100f);

    [Header("격자 한 칸의 간격 (단위: 미터)")]
    public float gridInterval = 10f;

    [Header("격자 선 색상 (개얇은 하얀선 고정)")]
    public Color gridColor = Color.white;

    private Material glMaterial;

    void CreateGLMaterial()
    {
        if (glMaterial == null)
        {
            // 유니티 자체의 단색 Render 통로 셰이더 호출
            Shader shader = Shader.Find("Hidden/Internal-Colored");
            glMaterial = new Material(shader);
            glMaterial.hideFlags = HideFlags.HideAndDontSave;
        }
    }

    void OnRenderObject()
    {
        Draw3DGrid();
    }

    // ── 💡 [버그 완벽 수정] 월드 변환 행렬을 GL 버퍼에 완전 결속하여 큐브 전체 동시 회전 ──
    void Draw3DGrid()
    {
        CreateGLMaterial();
        glMaterial.SetPass(0);

        GL.PushMatrix();
        
        // 💡 핵심 교정: 이 스크립트가 붙은 오브젝트의 Local matrix(이동, 회전, 스케일이 모두 포함됨)를 월드로 매핑!
        GL.MultMatrix(transform.localToWorldMatrix);
        
        GL.Begin(GL.LINES);
        GL.Color(gridColor);

        // 상자의 중심(0,0,0)을 기준으로 사방 여백 계산 (순수 로컬 공간)
        Vector3 half = stageSize * 0.5f;

        // 1. [바닥 및 천장 가로선] X축으로 뻗어나가는 선들을 로컬 축 기준으로 정렬
        for (float y = -half.y; y <= half.y; y += gridInterval)
        {
            for (float z = -half.z; z <= half.z; z += gridInterval)
            {
                GL.Vertex(new Vector3(-half.x, y, z));
                GL.Vertex(new Vector3(half.x, y, z));
            }
        }

        // 2. [사방 세로 기둥선] Y축으로 솟아오르는 기둥선 정렬
        for (float x = -half.x; x <= half.x; x += gridInterval)
        {
            for (float z = -half.z; z <= half.z; z += gridInterval)
            {
                GL.Vertex(new Vector3(x, -half.y, z));
                GL.Vertex(new Vector3(x, half.y, z));
            }
        }

        // 3. [바닥 및 종단 세로선] Z축 깊이 방향으로 관통하는 선 정렬
        for (float x = -half.x; x <= half.x; x += gridInterval)
        {
            for (float y = -half.y; y <= half.y; y += gridInterval)
            {
                GL.Vertex(new Vector3(x, y, -half.z));
                GL.Vertex(new Vector3(x, y, half.z));
            }
        }

        GL.End();
        GL.PopMatrix();
    }
}