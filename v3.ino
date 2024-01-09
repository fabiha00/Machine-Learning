#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <MPU6050.h>
#include <LiquidCrystal.h>
#include <math.h>
#include <SoftwareSerial.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 32
#define OLED_RESET 4
Adafruit_SSD1306 display(OLED_RESET);
LiquidCrystal lcd(12, 11, 4, 5, 6, 7); // initialize the LCD with the correct pins
SoftwareSerial sim900(9, 10); // RX, TX pins of SIM900


MPU6050 mpu;
float threshold = 1.1; // set threshold value in g (adjust as necessary)
int ledPin = 8;
int16_t ax, ay, az;

void setup() {
  Wire.begin();
  Serial.begin(9600);
  sim900.begin(9600); // initialize software serial for SIM900
  pinMode(ledPin, OUTPUT);
  mpu.initialize();
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  display.clearDisplay();
  display.setTextColor(WHITE);
  display.setTextSize(1);
  //display.setCursor(0,0);
 // display.println("Earthquake Detection");
  //display.display();
  lcd.begin(16, 2); // initialize the LCD with 16 columns and 2 rows
  lcd.clear(); // clear the LCD screen
  //lcd.print("Hello World!"); // write a message to the LCD
}

void sms(){
  sim900.println("AT+CMGF=1"); // set SMS mode to text
  delay(1000);
  sim900.println("AT+CMGS=\"+8801759209101\""); // replace with phone number to send SMS to
  delay(1000);
  sim900.println("Earthquake SOS!!!!"); // replace with message to send
  delay(1000);
  sim900.write(26); // send CTRL+Z to end SMS message
  delay(1000);
}


  unsigned long previousMillis = 0;
const int interval = 100; // Interval in milliseconds between each data point
int dataPoints[SCREEN_WIDTH]; // Array to store acceleration data points
int dataIndex = 0; // Current index in dataPoints array
int maxData = -10000; // Maximum value of dataPoints array
int minData = 10000; // Minimum value of dataPoints array


void setupGraph(int16_t ax, int16_t ay) {

   // Draw x-axis
  display.drawLine(0, SCREEN_HEIGHT / 2, SCREEN_WIDTH, SCREEN_HEIGHT / 2, WHITE);

  // Draw y-axis
  display.drawLine(0, 0, 0, SCREEN_HEIGHT, WHITE);

  // Draw data points
  int x = 0;
  for (int i = 0; i < SCREEN_WIDTH; i++) {
    int y = map(dataPoints[i], minData, maxData, SCREEN_HEIGHT, 0);
    display.drawPixel(x, y, WHITE);
    x++;
  }

  display.display();


}

void loop() {
  mpu.getAcceleration(&ax, &ay, &az);
  // Print the data to the serial monitor
  Serial.print("Accelerometer (mg): ");
  Serial.print(ax); Serial.print(", ");
  Serial.print(ay); Serial.print(", ");
  Serial.println(az);

  static unsigned long lastTime = 0;
  unsigned long currentTime = millis();


 

  float accelerationX = ax / 16384.0; // calculate acceleration in g
  float accelerationY = ay / 16384.0;
  float accelerationZ = az / 16384.0;
  float acceleration = sqrt(accelerationX * accelerationX + accelerationY * accelerationY + accelerationZ * accelerationZ);
  Serial.println(acceleration); 
  float convertedValue = map(acceleration, -32768, 32767, 0, SCREEN_HEIGHT);
  int graphHeight = SCREEN_HEIGHT - convertedValue;



 unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= interval) {

    // Store acceleration data in dataPoints array
    dataPoints[dataIndex] = (int)acceleration;

    // Update maxData and minData values
    if (acceleration > maxData) {
      maxData = (int)acceleration;
    } else if (acceleration < minData) {
      minData = (int)acceleration;
    }

    // Increment dataIndex and wrap around if necessary
    dataIndex++;
    if (dataIndex >= SCREEN_WIDTH) {
      dataIndex = 0;
      maxData = -10000;
      minData = 10000;
    }





  if (acceleration >= threshold) {
    digitalWrite(ledPin, HIGH); // turn on LED
    //lcd.print("Earthquake Detected!");
    display.clearDisplay();
    display.setCursor(0,0);
    //display.print("Earthquake Detected!");
    display.setCursor(0,10);
    //display.print("Acceleration: ");
    //display.print(acceleration, 2);
    //display.print(" g");
    setupGraph(ax, ay);
    display.display();
    sms();

    int num_bars = map(acceleration, 0, 1000, 0, 16);
    // display bar graph on LCD screen
    lcd.clear(); // clear the LCD screen
     lcd.print("Earthquake Detected!"); 
    lcd.setCursor(0, 1);
    lcd.print("A:"); 
    lcd.print(acceleration); 
    lcd.print("g");
  }
  else {
    digitalWrite(ledPin, LOW); // turn off LED
    //lcd.print("No Earthquake!");
    display.clearDisplay();
    display.setCursor(0,0);
    //display.println("No Earthquake");
    //setupGraph(ax, ay);
    

    int num_bars = map(acceleration, 0, 1000, 0, 16);
    // display bar graph on LCD screen
    lcd.clear(); // clear the LCD screen
    lcd.print("No Earthquake"); 
    lcd.setCursor(0, 1);
    lcd.print("A:"); 
    lcd.print(acceleration); 
    lcd.print("g");

    
  }
  delay(100);
  }
}
