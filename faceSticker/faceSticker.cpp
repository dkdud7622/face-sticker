#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\core.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <fstream>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>



using namespace dlib;
using namespace std;
using namespace cv;
string sticker_name = "sticker2.png";
Point input_point;
std::vector <Point2f> standard = { {24.4681,143.5106},{25.1489,175.4255},{ 29.0638,207.3404},{36.0426, 239.4044},{ 48.2979 ,268.8512}, {68.0426,294.7236}, {91.3617 , 317.3619}, { 117.9149, 335.064}, { 147.7021,340.0853}, {177.9574, 334.2766}, {204,315.8936},{ 227.7447 ,293.4255},{246.8936 ,266.9574},{ 259.0638, 236.8938 },{264.6809, 203.7874 },{267.1915 , 171.1917},{ 267.1915, 138.681},{ 47.4468,118.7447 },{61.8723, 106.4894},{82.2979,102.9149},{ 102.8511, 105.4255},{ 122.6383, 113.7234},{162.9787, 112.4894},{182.8936, 103.6809},{203.9574, 100.234},{224.1277, 102.7447},{239.9574, 114.3617},{143.3191, 137.2128 },{143.3191, 158.2766},{143.8298, 180.1064},{143.8298, 201.1702},{120.4681, 215.5957},{132.0851, 219.4255},{144.6809, 223.6809},{156.6809, 219.4255},{168.4255, 214.9574},{71.4468, 140.2766},{83.7872, 133.0426},{99.5319, 133.0426},{112.5532, 142.7872},{98.8936, 146.0213},{83.7872, 146.0213},{175.1702, 141.9576},{188.2553, 131.681},{203.1915, 130.9151},{216.4043, 138.3832},{204.5319, 143.9363},{189.4681, 143.9363},{98.6809, 253.9787},{115.8298, 247.1277},{132.8511, 243.6383},{145.0213, 246.1489},{158.2128, 242.9574},{176, 245.4681},{193.9574, 252.617},{177.4468, 269.9787},{160.2553, 278.4894},{145.7872, 280.4468},{132.8511, 279.2553},{115.8298, 271.766},{106.1277, 255.2979},{132.8511, 253.9787},{145.0213, 255.1277},{158.3771, 253.4448},{185.9516, 253.9787},{158.8085, 262.8085},{145.7872, 264.0198},{132.8511, 262.617} };
std::vector <Point2f> now;
Mat_<float> HomoMatrix;
Mat_<float> perspective;

class Sticker
{
public:
    Mat sticker;
    Mat Mask;
    Mat output_sticker;
    Mat output_Mask;
    Mat temp;
    void draw(Mat& image, const Mat& M);
    void maskdraw(Mat& image, const Mat& M);
    void load(const string& image, const string& mask);
    int get_width(const float x1, const float x2);
    int get_height(const float y1, const float y2);

private:

};

void Sticker::draw(Mat& image, const Mat& M)
{
    warpPerspective(sticker, output_sticker, M, image.size());
    warpPerspective(Mask, output_Mask, M, image.size(), 1, 0, Scalar(255, 255, 255));
    output_sticker.copyTo(image, 253 - output_Mask);
}
void Sticker::maskdraw(Mat& image, const Mat& M)
{
    warpPerspective(Mask, temp, M, image.size(), 1, 0, Scalar(255, 255, 255));
    temp.copyTo(image, 253 - temp);
}
void Sticker::load(const string& image, const string& mask)
{
    string image_path = "C:/Users/ayoung/Desktop/sticker/" + image;
    string mask_path = "C:/Users/ayoung/Desktop/sticker/" + mask;
    sticker = imread(image_path);
    Mask = imread(mask_path, cv::IMREAD_GRAYSCALE);
}
int Sticker::get_width(const float x1, const float x2)
{
    int width = abs(x1 - x2) * 1.4; //너비는 face landmarks 중 0번과 16번의 x좌표의 차이의 절대값.
    //너무 자잘자잘하게 변화해서 1의자리에서 반올림.
    width = (int)(ceil(width / 10.0) * 10);
    return width;
}

int Sticker::get_height(const float y1, const float y2)
{
    int height = abs(y1 - y2) * 2; //높이는 face landmarks 중 27번과 8번의 y좌표의 차이의 절대값.
    height = (int)(ceil(height / 10.0) * 10);
    return height;
}



void drawPoints(cv::Mat& image, full_object_detection landmarks)
{

    for (int i = 0; i < landmarks.num_parts(); i++) {
        //얼굴의 랜드마크들을 i로 순회함. 30번 : 코 중앙, landmarks들은 part로 표기.
        //cout <<"{" << landmarks.part(i).x() << "," << landmarks.part(i).y()<<"},";
        Point2f node = Point2f(landmarks.part(i).x(), landmarks.part(i).y());
        now.push_back(node);
        //cv::circle(image, cv::Point(landmarks.part(i).x(), landmarks.part(i).y()), 1, cv::Scalar(0, 0, 0), -1);
    }
    std::vector <int> temp(0);
    //cv::circle(image, cv::Point(landmarks.part(36).x(), landmarks.part(36).y()), 3, cv::Scalar(0, 255, 255), -1);

    HomoMatrix = findHomography(standard, now, LMEDS, 100.0, temp);
    now.clear();
    //cout << "끝" << endl;
}


void onMouseEvent(int event, int x, int y, int flags, void*)
{
    switch (event) {
    case EVENT_LBUTTONDOWN:

        input_point = Point(x, y);
        cout << "{" << input_point.x << "," << input_point.y << "},";
        if (input_point.x < 125) {
            sticker_name = "sticker1.png";
        }
        else if (input_point.x >= 125 && input_point.x < 255) {
            sticker_name = "sticker2.png";
        }
        else if (input_point.x >= 255 && input_point.x < 365) {
            sticker_name = "sticker3.png";
        }
        else if (input_point.x >= 365 && input_point.x < 500) {
            sticker_name = "sticker4.png";
        }
        else if (input_point.x >= 500) {
            sticker_name = "sticker5.png";
        }
        break;

    }

}


int main()
{
    Sticker st;
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cerr << "Unable to connect to camera" << endl;
        return 1;
    }

    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor pose_model;
    deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

    cv::Mat select = imread("C:/Users/ayoung/Desktop/sticker/select.png");
    cv::Mat face = imread("C:/Users/ayoung/Desktop/sticker/face.png");
    cv::Mat face_mask = imread("C:/Users/ayoung/Desktop/sticker/face.png", cv::IMREAD_GRAYSCALE);

    Mat st1 = imread("C:/Users/ayoung/Desktop/sticker/sticker1.png");
    Mat st2 = imread("C:/Users/ayoung/Desktop/sticker/sticker2.png");
    Mat st3 = imread("C:/Users/ayoung/Desktop/sticker/sticker3.png");
    Mat st4 = imread("C:/Users/ayoung/Desktop/sticker/sticker4.png");
    Mat st5 = imread("C:/Users/ayoung/Desktop/sticker/sticker5.png");
    Mat resize1, resize2, resize3, resize4, resize5;
    resize(st1, resize1, Size(st1.cols * 0.4, st1.rows * 0.4));
    resize(st2, resize2, Size(st1.cols * 0.4, st1.rows * 0.4));
    resize(st3, resize3, Size(st1.cols * 0.4, st1.rows * 0.4));
    resize(st4, resize4, Size(st1.cols * 0.4, st1.rows * 0.4));
    resize(st5, resize5, Size(st1.cols * 0.4, st1.rows * 0.4));

    Mat connect1, connect2, connect3, connect4;
    hconcat(resize1, resize2, connect1);
    hconcat(connect1, resize3, connect2);
    hconcat(connect2, resize4, connect3);
    hconcat(connect3, resize5, connect4);
    Mat canvas;
    Mat canvas2;
    while (true)
    {
        st.load(sticker_name, sticker_name);
        // Grab a frame
        cv::Mat temp;
        if (!cap.read(temp))
        {
            break;
        }
        cv::flip(temp, temp, 1);      //좌우반전 없애기
        cv_image<bgr_pixel> cimg(temp);
        Mat result1 = temp.clone();
        Mat result;
        result1.convertTo(result, -1, 1.2, 10);

        canvas = temp.clone();
        canvas = Scalar(255, 255, 255);
        canvas2 = canvas.clone();
        Mat canvas3 = canvas.clone();
        Mat canvas4 = canvas.clone();

        // Detect faces 
        std::vector<dlib::rectangle> faces = detector(cimg);
        // Find the pose of each face.
        std::vector<full_object_detection> shapes;
        for (unsigned long i = 0; i < faces.size(); ++i)
        {
            shapes.push_back(pose_model(cimg, faces[i]));
        }

        for (int i = 0; i < shapes.size(); i++) {
            if (shapes.size() == 0) break; //얼굴 인식 안되면 반복문 나가기
            drawPoints(result, shapes[i]);
            //;drawPoints(canvas2, shapes[i]);
            st.draw(canvas2, HomoMatrix);
            st.maskdraw(canvas3, HomoMatrix);
            st.draw(result, HomoMatrix);

        }

        cv::imshow("select", connect4);
        cv::setMouseCallback("select", onMouseEvent, (void*)&select);
        cv::imshow("result", result);
        if (waitKey(1) == 27) // 27 : esc

            break;

    }
}
