#!/bin/sh

g++ -O3 -std=c++17 -g \
      -Wall \
      -Wextra \
      -Wno-deprecated \
      -Wno-deprecated-declarations \
      -Wno-sign-compare \
      -Wno-unused \
      -Wunused-label \
      folly.cpp \
      -pthread \
      -lfollybenchmark \
      -lfolly \
      -lglog \
      -lgflags \
      -ldouble-conversion \
      -lboost_regex \
      -ldl \
      -liberty\
      -o folly.o
