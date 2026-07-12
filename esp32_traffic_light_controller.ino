/*
  esp32cam_stream_optimized.ino
  ---------------------
  STAGE 1 - CAMERA NODE ONLY (LOW LATENCY VERSION).
  Optimized specifically to reduce lag for OpenCV vehicle counting.

  Arduino IDE settings required:
    Board           : "AI Thinker ESP32-CAM"
    Partition Scheme: "Huge APP (3MB No OTA)"
    PSRAM           : Enabled
*/

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

// ---- EDIT THESE ----------------------------------------------------
const char* WIFI_SSID     = "FOE_Students";
const char* WIFI_PASSWORD = "FOE@30st";
// ---------------------------------------------------------------------

// AI-Thinker ESP32-CAM pin map
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

WebServer server(80);
WebServer altServer(81);

// ---------------------------------------------------------------------
void handleRoot(WebServer &webServer) {
  String html = "<html><body style='background:#111;color:#eee;text-align:center;font-family:sans-serif'>"
                "<h2>ESP32-CAM Live Feed</h2>"
                "<img src='/stream' style='max-width:95%;border:2px solid #555'>"
                "</body></html>";
  webServer.send(200, "text/html", html);
}

// Streams frames as multipart/x-mixed-replace (standard MJPEG-over-HTTP)
void handleStream(WebServer &webServer) {
  WiFiClient client = webServer.client();

  String header = "HTTP/1.1 200 OK\r\n";
  header += "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
  webServer.sendContent(header);

  while (client.connected()) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      delay(30); 
      continue;
    }

    char partHeader[64];
    int hlen = snprintf(partHeader, sizeof(partHeader),
                        "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
                        fb->len);

    webServer.sendContent(partHeader, hlen);
    client.write(fb->buf, fb->len);
    webServer.sendContent("\r\n");

    esp_camera_fb_return(fb);

    if (!client.connected()) break;
    
    // Tiny delay to yield to the ESP32 network stack tasks
    vTaskDelay(1 / portTICK_PERIOD_MS); 
  }
}

void setupStreamServer(WebServer &webServer) {
  webServer.on("/", HTTP_GET, [&webServer]() { handleRoot(webServer); });
  webServer.on("/stream", HTTP_GET, [&webServer]() { handleStream(webServer); });
  webServer.begin();
}

// ---------------------------------------------------------------------
bool setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk  = XCLK_GPIO_NUM;
  config.pin_pclk  = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href  = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn  = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  
  // Set to 10MHz to maintain frame transmission stability under heavy Wi-Fi load
  config.xclk_freq_hz = 10000000; 
  config.pixel_format  = PIXFORMAT_JPEG;

  if (psramFound()) {
    // CIF (400x296) provides low-latency frames while preserving edge lines for OpenCV
    config.frame_size   = FRAMESIZE_CIF;   
    config.jpeg_quality = 16;              // Increased compression (lower file sizes)
    config.fb_count     = 2;
  } else {
    config.frame_size   = FRAMESIZE_QVGA;  
    config.jpeg_quality = 18;
    config.fb_count     = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init FAILED, error 0x%x\n", err);
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  Serial.println();

  if (!setupCamera()) {
    Serial.println("Halting - fix camera init error above before continuing.");
    while (true) delay(1000);
  }

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  Serial.println();

  Serial.print("Camera Stream Ready! Go to: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/stream");
  Serial.print("Alternative URL: http://");
  Serial.print(WiFi.localIP());
  Serial.println(":81/stream");
  Serial.print("(Web preview page: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/)");

  setupStreamServer(server);
  setupStreamServer(altServer);
}

void loop() {
  server.handleClient();
  altServer.handleClient();
}