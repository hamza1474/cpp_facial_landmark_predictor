#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <dlib/array.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <iostream>
using namespace std;

int main()
{

	string predictor_path = "C:\\Users\\AmeerHamza\\source\\repos\\facial_landmark_dlib\\shape_predictor_68_face_landmarks.dat";
	// Initialize shape predictor and face detector
	dlib::shape_predictor predictor;
	dlib::deserialize(predictor_path) >> predictor;
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

	// Start the stream
	cv::VideoCapture vs;
	if (!vs.open(0)) {
		return 0;
	}
	while (true) {
		cv::Mat frame;
		//cv::Mat gray;
		vs >> frame;

		// Convert Frame to dlib readable format
		IplImage z_ipl = cvIplImage(frame);
		dlib::cv_image<dlib::bgr_pixel> dlib_img(z_ipl); 

		// Detect Faces
		std::vector<dlib::rectangle> faces = detector(dlib_img);

		// Display Number of Faces on stream
		cv::String text = "Number of faces: " + std::to_string(faces.size());
		cv::putText(frame, text, cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0, 255), 3); 

		// Initializer Shapes array
		std::vector<dlib::full_object_detection> shape_array;


		// Loop over detected faces
		for (int i = 0; i < faces.size(); i++)
		{
			// predict landmards and push to landmark array
			dlib::full_object_detection shape = predictor(dlib_img, faces[i]);
			shape_array.push_back(shape);

			// show bounding box on face
			dlib::rectangle r = shape.get_rect();
			cv::Rect rect = cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
			cv::rectangle(frame, rect,cv::Scalar(0, 255, 0, 255), 2);

			// Loop over the 68 facial landmarks
			for (uint i = 0; i < shape.num_parts(); i++)
			{
				// convert each landmark to cv readable point and
				// draw circle over it
				uint x = shape.part(i).x();
				uint y = shape.part(i).y();
				cv::circle(frame, cv::Point2i(x, y), 1, cv::Scalar(0, 0, 255, 255), -1);
			}

		}

		// Display the frame
		cv::imshow("Frame", frame);
		if (cv::waitKey(10) == 27) break;
	}

	
	return 0;
}