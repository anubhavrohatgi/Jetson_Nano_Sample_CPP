# Writing Your Own Object Recognition Code C++ on Jetson Nano

## Writing Your Own Object Recognition Code
Writing your own object recognition code isn’t actually that hard, and even in C++ can be done in a fairly compact manner if you’re not trying to do any complicated stuff around the task of classification.

You can grab the code and the associated build file from the command line using wget, and then build it as follows.

$ cd ~ <br />
$ mkdir object_recognition <br />
$ cd object_recognition <br />
$ wget https://gist.githubusercontent.com/aallan/4de3a74676d4ff10a476c2d6c20b9255/raw/818eb292805520a9fc01aaaee2f7a5692cdf1f92/object_recognition.cpp <br />
$ wget https://gist.githubusercontent.com/aallan/9945105f8ae2aed47d96e23adb8dddc1/raw/fef4e1249de9f4be6763e40cfcd8e1a7b92a40d4/CMakeLists.txt <br />
$ cmake .  <br />
$ make <br />
$ ./object_recognition polar.jpeg

imageNet -- loaded 1000 class info entries <br />
networks/bvlc_googlenet.caffemodel initialized. <br />
class 0279 - 0.018535  (Arctic fox, white fox, Alopex lagopus) <br />
class 0294 - 0.015127  (brown bear, bruin, Ursus arctos) <br />
class 0296 - 0.746165  (ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus) <br />
class 0342 - 0.017142  (wild boar, boar, Sus scrofa) <br />
class 0360 - 0.085703  (otter) <br />
image is recognized as 'ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus' (class #296) with 74.616531% confidence



