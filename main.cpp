#if _MSC_VER >= 1600
#pragma execution_character_set("utf-8")
#endif
#include<QFileDialog>
#include<QDebug>
#include <QApplication>
#include <QCoreApplication>
#include<QDebug>
#include<QDir>
#include<QFileDialog>
#include<QMainWindow>
#include<QObject>
#include<iostream>

//#include <fstream>
//#include <sstream>
//#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "yolo5.h"
#include<time.h>
using namespace cv;
using namespace std;



Net_config yolo_nets[5] = {
    {0.05, 0.2, 0.1, "best"},
    {0.5, 0.5, 0.5, "yolov5s"},
    {0.5, 0.5, 0.5,  "yolov5m"},
    {0.5, 0.5, 0.5, "yolov5l"},
    {0.5, 0.5, 0.5, "yolov5x"}
};
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    YOLO yolo_model(yolo_nets[0]);

    //从摄像头读人视频
    VideoCapture capture(0);
    //循环显示每一帧
    while(1)
    {
        Mat frame;//存储每一帧图像
        capture>>frame;//读取当前帧
        yolo_model.detect(frame);
        imshow("读取视频",frame);
        waitKey(1);//延时30毫秒
    }
}
