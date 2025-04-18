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
const char* ws_host = "192.168.31.122";
const int ws_port = 8000;
const char* ws_path = "/heart-rate/device-connect";

// System Settings
#define SYSTEM_BAUDRATE (115200)
#define WIFI_TIMEOUT    (5000)
#define RECONNECT_DELAY (2000)
#define DEVICE_ID "ESP32_HEARTRATE_PRESSURE_SENSOR"

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
    if (WiFi.status() != WL_CONNECTED) {
        connectToWiFi();
    }
}

void setup() {
    Serial.begin(SYSTEM_BAUDRATE);
    Wire.begin();
    bm1390glv.init();
    bm1390glv.start();
    connectToWiFi();
    setupWebSocket();
    Serial.println("[System] Setup completed");
}

void loop() {
    webSocket.loop();
    handleSensorData();
    reconnectIfNeeded();
}