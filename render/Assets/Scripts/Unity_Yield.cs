using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEngine.Networking;
using System.Text;

[System.Serializable]

public class location
{
    public string name;
    public int score;
}


public class Unity_Yield : MonoBehaviour
{
    // Start is called before the first frame update
   
    public string url = "";
    void Start()
    {
        Debug.Log("Start");

        location loc = new location();
        loc.name = "Bird_1";
        loc.score = 100;

        string json = JsonUtility.ToJson(loc);
        StartCoroutine(Sendlocation(json));
    }

    // Update is called once per frame
    IEnumerator Sendlocation(string json)
    {
        using (UnityWebRequest request = new UnityWebRequest(url, "POST"))
        {
            byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            // ✅ 추가되어야 할 코드
            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                Debug.Log("성공!");
            }
            else
            {
                Debug.LogError("실패: " + request.error);
            }
        
        }
    }
}

