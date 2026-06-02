using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using System.Text;


public class Unity_code : MonoBehaviour
{
    // Start is called before the first frame update

    private string url = "http://127.0.0.1:0000/update-score";
    void Start()
    {
        Debug.Log("started");
        StartCoroutine(Sendlocation("Bird_1", 100));
    }

    // Update is called once per frame
    IEnumerator Sendlocation(string name, int score)
    {
       string json = "{\"name\": \"" + name + "\", \"score\":" +score+"}";

       using (UnityWebRequest request = new UnityWebRequest(url, "POST"))
       {
            byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                Debug.Log("Score updated successfully!");
            }
            else
            {
                Debug.LogError("Error updating score: " + request.error);
                
            }
       }
    }
}

