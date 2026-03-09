Place your test.csv file in this 'data' directory.

When you build and upload the project using PlatformIO, the contents of this directory will be automatically packaged into a LittleFS filesystem image and flashed onto the ESP32's 'storage' partition.

The main.cpp firmware will then be able to open and read '/test.csv' from the onboard flash.
