using System;

[Serializable]
public class BirdInfo {
    public int id;
    public float lon;
    public float lat;
    public float direction;
    public float speed;
}

[Serializable]
public class EnvironmentData {
    public float daylight_hours;
    public float tailwind;
    public float humidity;
}

[Serializable]
public class FrameData {
    public int frame_id;
    public BirdInfo[] birds;
    public EnvironmentData environment;
}