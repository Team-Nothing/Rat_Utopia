#include <Arduino.h>
#include <Wire.h>
#include <BM1390GLV.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>

// WiFi Settings
const char* ssid = "LEDwifi2";     
const char* password = "fuuuuuuu";

// WebSocket Settings
const char* ws_host = "10.156.9.151";
const int ws_port = 8000;
const char* ws_path = "/heart-rate/device-connect";

// System Settings
#define SYSTEM_BAUDRATE (115200)
#define WIFI_TIMEOUT    (5000)
#define RECONNECT_DELAY (2000)
#define DEVICE_ID "ESP32_HEARTRATE_PRESSURE_SENSOR"

// D0 pin for WiFi control
#define PIN_D0 0  // 請根據您的 XIAO ESP32 確認 D0 的實際引腳
#define PIN_D1 1

// RTC 變數用於保存 WiFi 狀態 (在睡眠後仍能保持)
RTC_DATA_ATTR bool wifiEnabled = true;

BM1390GLV bm1390glv;
WebSocketsClient webSocket;
bool wsConnected = false;

unsigned long sensorIntervalMicros = 22700; // Default: 44100 Hz (1000000 / 44100)
unsigned long lastReadTimeMicros = 0;
unsigned long timeOffsetMicros = 0;

// Function Prototypes
void connectToWiFi();
void webSocketEvent(WStype_t type, uint8_t * payload, size_t length);
void setupWebSocket();
void handleSensorData();
void syncTime(unsigned long serverTimeMicros);
void reconnectIfNeeded();
void handlePingPong(const JsonObject& message);

// WebSocket Event Handler
void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
    if (type == WStype_DISCONNECTED) {
        wsConnected = false;
        Serial.println("[WebSocket] Disconnected");
    } else if (type == WStype_CONNECTED) {
        wsConnected = true;
        Serial.println("[WebSocket] Connected");
        // Send initial identity to server
        StaticJsonDocument<100> doc;
        doc["device_id"] = DEVICE_ID;
        String jsonString;
        serializeJson(doc, jsonString);
        webSocket.sendTXT(jsonString);
    } else if (type == WStype_TEXT) {
        StaticJsonDocument<200> doc;
        DeserializationError error = deserializeJson(doc, payload);
        if (!error) {
            if (doc.containsKey("type")) {
                const char* type = doc["type"];
                if (strcmp(type, "ping") == 0) {
                   handlePingPong(doc.as<JsonObject>());
                }
            }
            if (doc.containsKey("sample_rate")) {
                sensorIntervalMicros = 1000000 / doc["sample_rate"].as<int>();

                Serial.printf("[WebSocket] Sample rate updated: %lu Hz\n", doc["sample_rate"].as<int>());
            }
            if (doc.containsKey("server_time")) {
                syncTime(doc["server_time"]);
                Serial.println("[WebSocket] Time synchronized with server");
            }
        }
    }
}

// Handle Ping-Pong
void handlePingPong(const JsonObject& message) {
    StaticJsonDocument<100> doc;
    doc["type"] = "pong";
    String jsonString;
    serializeJson(doc, jsonString);
    webSocket.sendTXT(jsonString);
    Serial.println("[WebSocket] Pong sent");
}

// Sync Time with Server
void syncTime(unsigned long serverTimeMicros) {
    timeOffsetMicros = micros() - serverTimeMicros;
}

// Connect to WiFi
void connectToWiFi() {
    if (WiFi.status() == WL_CONNECTED) return;
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);
    Serial.printf("Connecting to WiFi: %s\n", ssid);

    unsigned long startAttemptTime = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - startAttemptTime < WIFI_TIMEOUT) {
        delay(100);
        Serial.print(".");
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n[WiFi] Connected");
    } else {
        Serial.println("\n[WiFi] Connection Failed");
    }
}

// Setup WebSocket
void setupWebSocket() {
    webSocket.begin(ws_host, ws_port, ws_path);
    webSocket.onEvent(webSocketEvent);
    webSocket.setReconnectInterval(RECONNECT_DELAY);
    Serial.println("[WebSocket] Initialized");
}

// Handle Sensor Data
void handleSensorData() {
    unsigned long now = micros();
    if (now - lastReadTimeMicros < sensorIntervalMicros) return; // Respect sample rate
    lastReadTimeMicros = now;

    int32_t result;
    float press, temp;
    result = bm1390glv.get_val(&press, &temp);
    if (result == BM1390GLV_COMM_OK && wsConnected) {
        StaticJsonDocument<200> doc;
        doc["type"] = "sensor_data";
        doc["t"] = temp;
        doc["p"] = press;
        doc["T"] = now - timeOffsetMicros; // Sync with server time
        String jsonString;
        serializeJson(doc, jsonString);
        webSocket.sendTXT(jsonString);
        // Serial.printf("[Sensor] Sent data: Temp=%.2f, Pressure=%.2f\n", temp, press);
    }
}

// Reconnect to WiFi if Needed
void reconnectIfNeeded() {
    if (WiFi.status() != WL_CONNECTED && wifiEnabled) {
        connectToWiFi();
    }
}

void setup() {
    Serial.begin(SYSTEM_BAUDRATE);
    // 設定 D0 為輸入，並啟用內部上拉電阻
    pinMode(PIN_D0, INPUT_PULLUP);
    pinMode(PIN_D1, OUTPUT);
    digitalWrite(PIN_D1, LOW);
    delay(100); // 短暫延遲以確保引腳狀態穩定
    
    // 檢查 D0 是否為接地狀態（LOW）
    if (digitalRead(PIN_D0) == LOW) {
        Serial.println("[System] D0 接地檢測到！永久關閉 WiFi 直到設備重啟");
        
        // 強制設置 WiFi 狀態為關閉
        wifiEnabled = false;
        
        Serial.println("[System] WiFi 已永久關閉，直到設備斷電重啟");
    }
    
    // 初始化 I2C 和感測器
    Wire.begin();
    bm1390glv.init();
    bm1390glv.start();
    
    // 根據 WiFi 狀態決定是否連接
    if (wifiEnabled) {
        Serial.println("正在連接");
        connectToWiFi();
        setupWebSocket();
    } else {
        Serial.println("等待~~~~~~~~~~~~~~~~~~~");
        WiFi.disconnect(true);
        WiFi.mode(WIFI_OFF);
    }
    
    Serial.println("[System] Setup completed");
}

void loop() {
    if (wifiEnabled) {
        webSocket.loop();
        reconnectIfNeeded();
    }
    handleSensorData();
}