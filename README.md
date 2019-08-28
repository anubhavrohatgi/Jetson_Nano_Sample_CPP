# Writing Your Own Object Recognition Code C++ on Jetson Nano

## Writing Your Own Object Recognition Code
Writing your own object recognition code isn’t actually that hard, and even in C++ can be done in a fairly compact manner if you’re not trying to do any complicated stuff around the task of classification.

You can grab the code and the associated build file from the command line using wget, and then build it as follows.

$ cd ~

$ mkdir object_recognition

$ wget https://gist.githubusercontent.com/aallan/4de3a74676d4ff10a476c2d6c20b9255/raw/818eb292805520a9fc01aaaee2f7a5692cdf1f92/object_recognition.cpp

$ wget https://gist.githubusercontent.com/aallan/9945105f8ae2aed47d96e23adb8dddc1/raw/fef4e1249de9f4be6763e40cfcd8e1a7b92a40d4/CMakeLists.txt

$ cmake .

$ make
