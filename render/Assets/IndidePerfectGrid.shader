Shader "Custom/InsidePerfectCubeGrid"
{
    Properties
    {
        _GridColor ("Grid Color", Color) = (1,1,1,1)
        _GridSize ("Grid Density (빽빽함)", Float) = 1.0
        _LineWidth ("Line Width (선 두께)", Range(0.001, 0.5)) = 0.05
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        LOD 100

        // 상자 겉면을 투명하게 지우고 오직 안쪽(내부벽, 바닥, 천장)만 렌더링
        Cull Front

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float3 localPos : TEXCOORD0;
                float3 localNormal : TEXCOORD1;
            };

            fixed4 _GridColor;
            float _GridSize;
            float _LineWidth;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.localPos = v.vertex.xyz;
                o.localNormal = abs(v.normal); 
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float3 pos = i.localPos;
                float3 absN = normalize(i.localNormal);
                
                // 💡 [비율 불균형 완벽 조치 핵심]
                // 박스의 높이(Y축)가 가로·깊이(100)의 절반인 50이므로,
                // Y축 좌표를 사용하는 계산 포인터에 0.5를 곱해 기하학적 스케일을 1:1로 리밸런싱합니다.
                float3 scaledPos = float3(pos.x, pos.y * 0.5, pos.z);
                
                // 보정된 scaledPos 데이터를 기반으로 삼면 투영 격자 무늬 연산 수행
                float2 gridX = abs(frac(float2(pos.y * 0.5, pos.z) * _GridSize) - 0.5);
                float2 gridY = abs(frac(float2(pos.x, pos.z) * _GridSize) - 0.5);
                float2 gridZ = abs(frac(float2(pos.x, pos.y * 0.5) * _GridSize) - 0.5);
                
                // 선의 굵기를 제어하는 필터 연산
                float2 lineCheckX = step(gridX, _LineWidth);
                float2 lineCheckY = step(gridY, _LineWidth);
                float2 lineCheckZ = step(gridZ, _LineWidth);
                
                // 각 면의 방향 가중치에 맞춰 조립
                float finalLine = max(lineCheckX.x, lineCheckX.y) * absN.x +
                                  max(lineCheckY.x, lineCheckY.y) * absN.y +
                                  max(lineCheckZ.x, lineCheckZ.y) * absN.z;

                // 선이 그려지지 않는 빈 공간은 암전 블랙으로 밀어버립니다.
                if(finalLine <= 0.05)
                {
                    return fixed4(0.0, 0.0, 0.0, 1.0); // 배경 검은색 고정
                }

                return _GridColor; // 격자 하얀 선 출력
            }
            ENDCG
        }
    }
}