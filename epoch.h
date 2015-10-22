#ifndef AM_RT_EPOCH_H
#define AM_RT_EPOCH_H

#include <stdlib.h>
#include <time.h>
class epoch{
 public:
  time_t t1, t2;

  epoch() {}

  void start() {
    time(&t1);
  }

  int elapsed() {
    time(&t2);
    return difftime(t2,t1);

  }
};

//class epoch{
// public:
//  struct timeval t1, t2;
//
//  epoch() {}
//
//  void start() {
//    gettimeofday(&t1,NULL);
//  }
//
//  int elapsed() {
//    gettimeofday(&t2,NULL);
//    return t2.tv_sec - t1.tv_sec;
//
//  }
//};

#endif //AM_RT_EPOCH_H
