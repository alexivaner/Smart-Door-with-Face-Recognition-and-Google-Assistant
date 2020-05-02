#include <Servo.h>

Servo doorlock;
int inpin = 4;
int suara = 5;
int val;

void setup() {
  doorlock.attach(3);
  pinMode(inpin, INPUT);
  pinMode(suara, OUTPUT);
}

void loop() {
  val = digitalRead(inpin);
  if (val == HIGH) {
    doorlock.write(0);    // sets the servo position according to the scaled value
    while (digitalRead(inpin) == HIGH)
    {
      digitalWrite(suara, HIGH);
      delay(200);
      digitalWrite(suara, LOW);
      delay(150);
    }
  }
  else
  {
    doorlock.write(155);                  // sets the servo position according to the scaled value
    digitalWrite(suara, LOW);
  }
}
