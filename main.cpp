// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include <getopt.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>

#include "inference.h"

#include <sys/stat.h>
#include <sys/types.h>


using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    std::string projectBasePath = "/home/user/Eden_yolov8_C++/ultralytics/examples/YOLOv8-CPP-Inference"; // Set your ultralytics base path
    std::cout << cv::getBuildInformation() << std::endl;
    
    bool runOnGPU = true;

    //
    // Pass in either:
    //
    // "yolov8s.onnx" or "yolov5s.onnx"
    //
    // To run Inference with yolov8/yolov5 (ONNX)
    //

    // Note that in this example the classes are hard-coded and 'classes.txt' is a place holder.
    Inference inf(projectBasePath + "/best_LPR.onnx", cv::Size(640, 640), "/classes.txt", runOnGPU);

    // 設定 output_data 路徑
    std::string outputDir = projectBasePath + "/output_data";

    // 建立 output_data 資料夾（如果不存在的話）
    DIR* testDir = opendir(outputDir.c_str());
    if (testDir) {
        std::cout << "資料夾已存在：" << outputDir << std::endl;
        closedir(testDir);
    } else {
        if (mkdir(outputDir.c_str(), 0755) == 0) {
            std::cout << "成功建立資料夾：" << outputDir << std::endl;
        } else {
            std::cerr << "建立資料夾失敗：" << strerror(errno) << std::endl;
        }
    }    


    std::vector<std::string> imageNames;
    std::string imageFolder = projectBasePath + "/data";

    DIR* dir;
    struct dirent* ent;
    
    if ((dir = opendir(imageFolder.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string fileName = ent->d_name;
            if (fileName.size() > 4 && 
                (fileName.substr(fileName.size() - 4) == ".jpg" || 
                 fileName.substr(fileName.size() - 4) == ".JPG")) {
                imageNames.push_back(imageFolder + "/" + fileName);
            }
        }
        closedir(dir);
    }

    for (int i = 0; i < imageNames.size(); ++i)
    {

        cv::Mat frame = cv::imread(imageNames[i]);
        if (frame.empty()) {
            std::cerr << "Image not found or empty!" << std::endl;
            continue;
        }

        std::cerr << "Image found: ["<< i <<"]" << std::endl;

        // Inference starts here...
        std::vector<Detection> output = inf.runInference(frame);
        int detections = output.size();
        std::cout << "Number of detections:" << detections << std::endl;

        for (int j = 0; j < detections; ++j)
        {
            Detection detection = output[j];

            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;

            // Detection box
            // cv::rectangle(frame, box, color, 2);
            std::cout << "Detection " << j << ": " << box << std::endl;
            std::cout << "box: " << box << std::endl;
            
            cv::Rect safebox = box & cv::Rect(0, 0, frame.cols, frame.rows);
            cv::Mat detected_img = frame(safebox).clone();
            
            //std::string(imageNames[i])
            size_t datadir_Pos = imageNames[i].find_last_of('/');
            std::string datadir = imageNames[i].substr(0, datadir_Pos);
            size_t outputdir_Pos = datadir.find_last_of('/');
            std::string outputdir = datadir.substr(0, outputdir_Pos)+"/output_data";

            size_t dataName_dotPos = imageNames[i].find_last_of('.');
            std::string dataName = imageNames[i].substr(datadir_Pos, (dataName_dotPos-datadir_Pos));

            std::string output_filename = outputdir + dataName + '-' + std::to_string(j) + ".jpg";
            std::cout << output_filename << std::endl;
            cv::imwrite(output_filename, detected_img);
            
            // Detection box text
            // std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            // cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            // cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            // cv::rectangle(frame, textBox, color, cv::FILLED);
            // cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        }
        // Inference ends here...

        // This is only for preview purposes
        // float scale = 0.8;
        // cv::resize(frame, frame, cv::Size(frame.cols*scale, frame.rows*scale));
        // cv::imshow("Inference", frame);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Total execution time: " << elapsed.count() << " seconds." << std::endl;
    return 0;
}
