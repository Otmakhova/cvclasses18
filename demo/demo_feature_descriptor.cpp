/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

int demo_feature_descriptor(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);

    cv::Mat frame, frame_gray;
    auto detector = cvlib::corner_detector_fast::create();
    //auto detector_b = cv::KAZE::create();
    std::vector<cv::KeyPoint> corners;
    cv::Mat descriptors;
    utils::fps_counter fps;
    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;
        cv::imshow(main_wnd, frame);

        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
        detector->detect(frame, corners); // \todo use your detector (detector_b)
        cv::drawKeypoints(frame, corners, frame, cv::Scalar(0, 0, 255));

        utils::put_fps_text(frame, fps);
        // \todo add count of the detected corners at the top left corner of the image. Use green text color.
        utils::put_number_of_keypoints(frame, corners.size());
		cv::imshow(demo_wnd, frame);

    }
    detector->clear();
    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
